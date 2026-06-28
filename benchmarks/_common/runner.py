"""Training/evaluation runner — one ResultRow per seed."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.model_builder import build_model
from benchmarks._common.results import ResultRow
from benchmarks._common.seeds import seed_everything

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle
    from benchmarks._common.config import BenchmarkConfig


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
    for seed in cfg.seeds:
        seed_everything(cfg.backend, seed)
        model = build_model(cfg, bundle, seed=seed)
        if cfg.backend == "torch":
            epochs_run = _train_torch(model, cfg, bundle)
        elif cfg.backend == "jax":
            model, epochs_run = _train_jax(model, cfg, bundle, seed)
        elif cfg.backend == "keras":
            epochs_run = _train_keras(model, cfg, bundle)
        else:
            raise ValueError(f"Unknown backend: {cfg.backend!r}")

        scores = _evaluate(model, cfg, bundle)
        rows.append(
            ResultRow(
                dataset=cfg.dataset,
                backend=cfg.backend,
                mode=cfg.mode,
                residual=cfg.residual,
                seed=seed,
                scores=scores,
                epochs_run=epochs_run,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# PyTorch training loop
# ---------------------------------------------------------------------------


def _train_torch(model: Any, cfg: BenchmarkConfig, bundle: DatasetBundle) -> int:
    """Train a torch model in-place and return epochs completed.

    :param model: ``nn.Module`` returned by :func:`build_model`.
    :param cfg: Benchmark configuration.
    :param bundle: Dataset bundle providing training data.
    :returns: Number of epochs completed.
    """
    import torch
    import torch.nn as nn

    binary = bundle.task == "binary_classification"
    criterion: nn.Module = nn.BCELoss() if binary else nn.MSELoss()
    adam = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    x_train = torch.tensor(bundle.X_train, dtype=torch.float64)
    y_train = torch.tensor(bundle.y_train, dtype=torch.float64).unsqueeze(1)

    n = x_train.shape[0]
    batch_size = min(cfg.batch_size, n)
    lr = cfg.optimizer.lr

    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n)
        x_shuf = x_train[perm]
        y_shuf = y_train[perm]

        for start in range(0, n, batch_size):
            xb = x_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            adam.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            adam.step()

        # per-epoch LR decay
        if cfg.lr_decay is not None:
            lr = lr * cfg.lr_decay
            for pg in adam.param_groups:
                pg["lr"] = lr

    model.eval()
    return cfg.epochs


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

    @nnx.jit  # type: ignore[misc]
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

    scores: dict[str, float] = {}
    mse_val: float | None = None

    for metric in cfg.metrics:
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
            preds_bin = (y_pred >= 0.5).astype(np.float64)
            scores["accuracy"] = float(np.mean(preds_bin == y_true))
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

        with torch.no_grad():
            x_t = torch.tensor(bundle.X_test, dtype=torch.float64)
            out_np: np.ndarray = model(x_t).numpy().ravel()  # type: ignore[type-arg]
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
