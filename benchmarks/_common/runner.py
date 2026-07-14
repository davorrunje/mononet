"""Training/evaluation runner — one ResultRow per seed."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.model_builder import build_model
from benchmarks._common.results import ResultRow
from benchmarks._common.seeds import seed_everything

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle
    from benchmarks._common.config import BenchmarkConfig


def _torch_device() -> Any:
    """Select the torch device for benchmark training/eval.

    Uses ``$MONONET_TORCH_DEVICE`` when set (e.g. ``"cuda:0"`` / ``"cpu"``),
    otherwise CUDA when available, else CPU.

    :returns: A ``torch.device``.
    """
    import torch

    override = os.environ.get("MONONET_TORCH_DEVICE")
    if override:
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Divergence
# ---------------------------------------------------------------------------


def is_diverged(final_loss: float, baseline: float) -> bool:
    """Whether a run's final loss signals divergence.

    A run diverged if its final loss is non-finite or exceeds ``10x`` the
    predict-the-mean baseline (regression: ``Var[y]``; classification:
    base-rate binary cross-entropy).

    :param final_loss: Final loss of the trained model.
    :param baseline: Predict-the-mean baseline loss.
    :returns: ``True`` if the run diverged.
    """
    return (not math.isfinite(final_loss)) or (final_loss > 10.0 * baseline)


def _loss_and_baseline(
    y_pred: np.ndarray,  # type: ignore[type-arg]
    y_true: np.ndarray,  # type: ignore[type-arg]
    *,
    binary: bool,
) -> tuple[float, float]:
    """Compute the final loss and the predict-the-mean baseline.

    :param y_pred: Model predictions (probability-scale for classification).
    :param y_true: Ground-truth targets.
    :param binary: Whether the task is binary classification.
    :returns: ``(final_loss, baseline)``. For classification both are binary
        cross-entropy (model vs base-rate predictor); for regression both are
        MSE (model vs predict-the-mean, i.e. ``Var[y]``).
    """
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    if binary:
        eps = 1e-8
        p = np.clip(yp, eps, 1 - eps)
        loss = float(np.mean(-(yt * np.log(p) + (1 - yt) * np.log(1 - p))))
        base_rate = float(np.clip(np.mean(yt), eps, 1 - eps))
        baseline = float(
            -(base_rate * np.log(base_rate) + (1 - base_rate) * np.log(1 - base_rate))
        )
    else:
        loss = float(np.mean((yp - yt) ** 2))
        baseline = float(np.var(yt))
    return loss, baseline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run(cfg: BenchmarkConfig, bundle: DatasetBundle) -> list[ResultRow]:
    """Train and evaluate for each seed in *cfg.seeds*.

    :param cfg: Complete benchmark configuration.
    :param bundle: Preprocessed dataset bundle.
    :returns: One :class:`~benchmarks._common.results.ResultRow` per seed.
    """
    rows: list[ResultRow] = []
    binary = bundle.task == "binary_classification"
    for seed in cfg.seeds:
        seed_everything(cfg.backend, seed)
        model = build_model(cfg, bundle, seed=seed)
        train_diverged = False
        if cfg.backend == "torch":
            epochs_run, train_diverged = _train_torch(model, cfg, bundle)
        elif cfg.backend == "jax":
            model, epochs_run = _train_jax(model, cfg, bundle, seed)
        elif cfg.backend == "keras":
            epochs_run = _train_keras(model, cfg, bundle)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend!r}")

        y_pred = _predict(model, cfg, bundle)
        scores = _score_predictions(
            y_pred, bundle.y_test, binary=binary, metrics=cfg.metrics
        )
        final_loss, baseline = _loss_and_baseline(y_pred, bundle.y_test, binary=binary)
        diverged = train_diverged or is_diverged(final_loss, baseline)
        rows.append(
            ResultRow(
                dataset=cfg.dataset,
                backend=cfg.backend,
                mode=cfg.mode,
                residual=cfg.residual,
                seed=seed,
                scores=scores,
                epochs_run=epochs_run,
                diverged=diverged,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------


def _carve_val_split(x_all: Any, y_all: Any) -> tuple[Any, Any, Any, Any]:
    """Carve a 20% validation split for early stopping.

    :param x_all: All training inputs.
    :param y_all: All training targets, shape ``(N, 1)``.
    :returns: ``(x_train, y_train, x_val, y_val)``.
    """
    import torch

    n_all = x_all.shape[0]
    n_val = max(1, int(0.2 * n_all))
    split = torch.randperm(n_all, device=x_all.device)
    val_idx, tr_idx = split[:n_val], split[n_val:]
    return x_all[tr_idx], y_all[tr_idx], x_all[val_idx], y_all[val_idx]


def _torch_epoch(
    model: Any, adam: Any, criterion: Any, x_train: Any, y_train: Any, batch_size: int
) -> None:
    """Run one shuffled mini-batch epoch of in-place SGD.

    :param model: The torch module being trained.
    :param adam: The optimizer.
    :param criterion: The loss module.
    :param x_train: Training inputs.
    :param y_train: Training targets.
    :param batch_size: Mini-batch size.
    """
    import torch

    n = x_train.shape[0]
    perm = torch.randperm(n, device=x_train.device)
    x_shuf, y_shuf = x_train[perm], y_train[perm]
    for start in range(0, n, batch_size):
        xb = x_shuf[start : start + batch_size]
        yb = y_shuf[start : start + batch_size]
        adam.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        adam.step()


class _EarlyStop:
    """Best-epoch tracker + divergence detector for the torch training loop.

    ``diverged`` fires only on a **non-finite** validation loss (a genuine
    blow-up). It deliberately does *not* fire on a merely-large finite loss: an
    untrained model early in training is normally well above the predict-the-mean
    baseline, so a magnitude test here would false-positive on every run that
    later converges. Whether the *final* model is worse than ``10x`` the baseline
    is decided by the caller from the restored-best model's eval
    (:func:`is_diverged`), which is the plan's "final loss" definition.

    :param patience: Epochs without validation improvement before stopping.
    :param min_delta: Minimum *relative* improvement to reset patience — an epoch
        improves only when ``val_loss < best * (1 - min_delta)``. ``0.0`` reduces
        to "any improvement counts" (which on a slowly-improving regression loss
        never triggers early stopping).
    """

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = math.inf
        self.best_epoch = 0
        self.best_state: dict[str, Any] | None = None
        self.diverged = False
        self.no_improve = 0

    def _improved(self, val_loss: float) -> bool:
        """Whether ``val_loss`` clears the relative ``min_delta`` threshold.

        :param val_loss: Candidate validation loss.
        :returns: ``True`` if it is a sufficient improvement over the best so far.
        """
        if not math.isfinite(self.best_val):
            return True  # first finite loss always improves on inf
        return val_loss < self.best_val * (1.0 - self.min_delta)

    def update(self, model: Any, epoch: int, val_loss: float) -> bool:
        """Record ``val_loss`` for ``epoch``; return whether to stop training.

        :param model: The module being trained (its ``state_dict`` is snapshotted
            when the validation loss improves).
        :param epoch: 1-based epoch number just completed.
        :param val_loss: Validation loss for this epoch.
        :returns: ``True`` if training should stop (non-finite loss, or patience
            exhausted).
        """
        import copy

        if not math.isfinite(val_loss):
            self.diverged = True
            return True  # no point continuing a blown-up run
        if self._improved(val_loss):
            self.best_val, self.best_epoch = val_loss, epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.no_improve = 0
            return False
        self.no_improve += 1
        return self.no_improve >= self.patience


def _train_torch(
    model: Any, cfg: BenchmarkConfig, bundle: DatasetBundle
) -> tuple[int, bool]:
    """Train a torch model in-place; return ``(epochs_run, diverged)``.

    When ``cfg.early_stopping is None`` (every existing benchmark), training is
    the historical fixed-``cfg.epochs`` loop over all of ``X_train`` — behaviour
    is byte-for-byte unchanged, ``epochs_run == cfg.epochs`` and ``diverged`` is
    reported from the final-model eval by the caller.

    When ``cfg.early_stopping`` is set (the flavor ablation), a validation split
    is carved from ``X_train``: each epoch's validation loss is monitored, the
    best-epoch weights are restored at the end, ``epochs_run`` is the
    epochs-to-best, and ``diverged`` is ``True`` if any epoch's validation loss
    was non-finite (a genuine blow-up, which best-weight restore cannot mask).
    Whether the final model is merely worse than the baseline is decided by the
    caller from the eval (see :func:`is_diverged`).

    :param model: ``nn.Module`` returned by :func:`build_model`.
    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle providing training data.
    :returns: Tuple of (epochs run, diverged flag).
    """
    import torch
    import torch.nn as nn

    device = _torch_device()
    model.to(device)

    binary = bundle.task == "binary_classification"
    criterion: nn.Module = nn.BCELoss() if binary else nn.MSELoss()
    adam = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    x_all = torch.tensor(bundle.X_train, dtype=torch.float32, device=device)
    y_all = torch.tensor(bundle.y_train, dtype=torch.float32, device=device).unsqueeze(
        1
    )

    es = cfg.early_stopping
    if es is not None:
        x_train, y_train, x_val, y_val = _carve_val_split(x_all, y_all)
    else:
        x_train, y_train, x_val, y_val = x_all, y_all, None, None

    batch_size = min(cfg.batch_size, x_train.shape[0])
    lr = cfg.optimizer.lr
    stopper = _EarlyStop(es.patience, es.min_delta) if es is not None else None
    epochs_done = 0

    model.train()
    for epoch in range(cfg.epochs):
        epochs_done = epoch + 1
        _torch_epoch(model, adam, criterion, x_train, y_train, batch_size)

        # per-epoch LR decay
        if cfg.lr_decay is not None:
            lr = lr * cfg.lr_decay
            for pg in adam.param_groups:
                pg["lr"] = lr

        if stopper is None:
            continue

        # --- early stopping on the validation split ---
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val), y_val).item())
        model.train()
        if stopper.update(model, epochs_done, val_loss):
            break

    model.eval()
    if stopper is None:
        return epochs_done, False
    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        return stopper.best_epoch, stopper.diverged
    return epochs_done, stopper.diverged


# ---------------------------------------------------------------------------
# JAX / Flax NNX training loop
# ---------------------------------------------------------------------------


def _train_jax(
    model: Any, cfg: BenchmarkConfig, bundle: DatasetBundle, seed: int = 0
) -> tuple[Any, int]:
    """Train a Flax NNX model and return (updated model, epochs completed).

    :param model: Flax NNX module returned by :func:`build_model`.
    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle providing training data.
    :param seed: Per-run random seed used for minibatch shuffling.
    :returns: Tuple of (trained model, epochs run).
    """
    import jax.numpy as jnp
    import optax
    from flax import nnx

    binary = bundle.task == "binary_classification"

    # nnx.ModelAndOptimizer bundles model + optimizer state in one stateful object.
    # (nnx.Optimizer since Flax 0.11.0 no longer exposes .model — use ModelAndOptimizer.)
    mopt: Any = nnx.ModelAndOptimizer(model, optax.adam(cfg.optimizer.lr))

    x_train = jnp.array(bundle.X_train, dtype=jnp.float32)
    y_train = jnp.array(bundle.y_train, dtype=jnp.float32).reshape(-1, 1)

    n = x_train.shape[0]
    batch_size = min(cfg.batch_size, n)

    rng = np.random.default_rng(seed)

    def loss_fn(m: Any, xb: Any, yb: Any) -> Any:
        pred = m(xb)
        if binary:
            return jnp.mean(
                -(yb * jnp.log(pred + 1e-8) + (1 - yb) * jnp.log(1 - pred + 1e-8))
            )
        return jnp.mean((pred - yb) ** 2)

    @nnx.jit  # type: ignore[misc, untyped-decorator]
    def train_step(mo: Any, xb: Any, yb: Any) -> Any:
        loss, grads = nnx.value_and_grad(loss_fn)(mo.model, xb, yb)
        mo.update(grads)
        return loss

    lr = cfg.optimizer.lr
    for _ in range(cfg.epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]
            train_step(mopt, xb, yb)

        # per-epoch LR decay: rebuild with new lr
        if cfg.lr_decay is not None:
            lr = lr * cfg.lr_decay
            mopt = nnx.ModelAndOptimizer(mopt.model, optax.adam(lr))

    return mopt.model, cfg.epochs


# ---------------------------------------------------------------------------
# Keras training loop
# ---------------------------------------------------------------------------


def _train_keras(model: Any, cfg: BenchmarkConfig, bundle: DatasetBundle) -> int:
    """Train a Keras model via ``compile/fit`` and return epochs completed.

    :param model: ``keras.Model`` returned by :func:`build_model`.
    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle providing training data.
    :returns: Number of epochs completed.
    """
    import keras

    binary = bundle.task == "binary_classification"
    loss = "binary_crossentropy" if binary else "mse"

    lr_schedule: Any
    if cfg.lr_decay is not None:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=cfg.optimizer.lr,
            decay_steps=1,
            decay_rate=cfg.lr_decay,
        )
    else:
        lr_schedule = cfg.optimizer.lr

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=loss,
    )
    model.fit(
        bundle.X_train,
        bundle.y_train,
        batch_size=min(cfg.batch_size, len(bundle.X_train)),
        epochs=cfg.epochs,
        verbose=0,  # type: ignore[arg-type]
    )
    return cfg.epochs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model: Any, cfg: BenchmarkConfig, bundle: DatasetBundle
) -> dict[str, float]:
    """Compute requested metrics on the test split.

    :param model: Trained model (backend-native).
    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle providing test data.
    :returns: Dict mapping metric name to scalar value.
    """
    y_pred = _predict(model, cfg, bundle)
    y_true = bundle.y_test
    binary = bundle.task == "binary_classification"
    return _score_predictions(y_pred, y_true, binary=binary, metrics=cfg.metrics)


def _score_predictions(
    y_pred: np.ndarray,  # type: ignore[type-arg]
    y_true: np.ndarray,  # type: ignore[type-arg]
    *,
    binary: bool,
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """Compute the requested metrics from predictions and targets.

    :param y_pred: Model predictions (probability-scale for classification).
    :param y_true: Ground-truth targets.
    :param binary: Whether the task is binary classification.
    :param metrics: Metric names to compute.
    :returns: Dict mapping metric name to scalar value.
    :raises ValueError: If a classification-only metric is requested for a
        non-binary task, or an unknown metric name is given.
    """
    scores: dict[str, float] = {}
    mse_val: float | None = None
    for metric in metrics:
        if metric == "mse":
            mse_val = float(np.mean((y_pred - y_true) ** 2))
            scores["mse"] = mse_val
        elif metric == "rmse":
            if mse_val is None:
                mse_val = float(np.mean((y_pred - y_true) ** 2))
            scores["rmse"] = math.sqrt(mse_val)
        elif metric == "accuracy":
            if not binary:
                raise ValueError("accuracy metric requires binary_classification task")
            scores["accuracy"] = float(
                np.mean((y_pred >= 0.5).astype(np.float64) == y_true)
            )
        elif metric == "roc_auc":
            if not binary:
                raise ValueError("roc_auc metric requires binary_classification task")
            from sklearn.metrics import roc_auc_score

            scores["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
    return scores


def _predict(model: Any, cfg: BenchmarkConfig, bundle: DatasetBundle) -> np.ndarray:  # type: ignore[type-arg]
    """Run inference and return a 1-D NumPy array of predictions.

    :param model: Trained backend-native model.
    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle providing test data.
    :returns: 1-D float64 array of shape ``(n_test,)``.
    """
    if cfg.backend == "torch":
        import torch

        device = next(model.parameters()).device
        with torch.no_grad():
            x_t = torch.tensor(bundle.X_test, dtype=torch.float32, device=device)
            out_np: np.ndarray = model(x_t).cpu().numpy().ravel()  # type: ignore[type-arg]
        return out_np.astype(np.float64)

    if cfg.backend == "jax":
        import jax.numpy as jnp

        x_j = jnp.array(bundle.X_test, dtype=jnp.float32)
        out_np = np.array(model(x_j)).ravel()
        return out_np.astype(np.float64)

    if cfg.backend == "keras":
        raw = model.predict(bundle.X_test, verbose=0)  # type: ignore[arg-type]
        return np.array(raw, dtype=np.float64).ravel()

    raise ValueError(f"Unknown backend: {cfg.backend!r}")
