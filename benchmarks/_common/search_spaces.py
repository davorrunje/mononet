"""Per-flavor Optuna search space producing a BenchmarkConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec

if TYPE_CHECKING:
    import optuna

# Train-set-size threshold (rows) above which small batches make 50-epoch
# training intractable; the models are tiny so tuning is launch-bound, not
# capacity-bound. Derived from the loaded n_train so new datasets band
# automatically (no hand-maintained name set).
_LARGE_BATCH_THRESHOLD = 20_000
_BATCH_SIZES_SMALL = [8, 16, 32, 64, 128, 256]
_BATCH_SIZES_LARGE = [512, 1024, 2048, 4096]


def suggest_config(
    trial: optuna.Trial,
    *,
    dataset: str,
    backend: Literal["torch", "jax", "keras"],
    mode: Literal["switch", "absolute"],
    residual: bool,
    epochs: int,
    metric: Literal["accuracy", "rmse", "mse"],
    n_train: int,
    deep: bool = False,
) -> BenchmarkConfig:
    """Sample a BenchmarkConfig for one (dataset, flavor) trial.

    `convex_fraction` is searched only for absolute mode; switch keeps 0.5.
    `activation` is fixed to "elu" in Phase 2a.

    :param trial: Optuna trial used to suggest hyperparameter values.
    :param dataset: Dataset name (labels the config).
    :param backend: ML backend to target.
    :param mode: Monotonicity mode (`"absolute"` or `"switch"`).
    :param residual: Whether to use residual connections.
    :param epochs: Number of training epochs per trial.
    :param metric: Primary metric; propagated into `cfg.metrics` so the
        objective's metric and the training config always agree.
    :param n_train: Number of rows in the training set; selects the ``batch_size``
        band (large-batch band if ``n_train >= _LARGE_BATCH_THRESHOLD``).
    :param deep: When ``True``, draw ``depth`` from the deep categorical band
        ``{6, 10, 16}`` (residual skips make these trainable); otherwise draw
        ``depth`` from the shallow range ``[1, 4]``. Only affects the ``depth``
        dimension; all other hyperparameters are sampled identically.
    :returns: A fully populated `BenchmarkConfig` ready for `run()`.
    """
    width = trial.suggest_categorical("width", [8, 16, 21, 32, 64])
    if deep:
        depth = trial.suggest_categorical("depth", [6, 10, 16])
    else:
        depth = trial.suggest_int("depth", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr_decay = trial.suggest_float("lr_decay", 0.85, 1.0)
    batch_choices = (
        _BATCH_SIZES_LARGE if n_train >= _LARGE_BATCH_THRESHOLD else _BATCH_SIZES_SMALL
    )
    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    convex_fraction = (
        trial.suggest_float("convex_fraction", 0.0, 1.0) if mode == "absolute" else 0.5
    )
    return BenchmarkConfig(
        dataset=dataset,
        backend=backend,
        mode=mode,
        residual=residual,
        depth=depth,
        width=int(width),
        activation="elu",
        convex_fraction=convex_fraction,
        embed_hidden=(int(width),),
        dropout=dropout,
        optimizer=OptimizerSpec("adam", lr, weight_decay),
        lr_decay=lr_decay,
        batch_size=int(batch_size),
        epochs=epochs,
        early_stopping=None,
        seeds=(0,),
        metrics=(metric,),
    )
