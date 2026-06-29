"""Per-flavor Optuna search space producing a BenchmarkConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec

if TYPE_CHECKING:
    import optuna


def suggest_config(
    trial: optuna.Trial,
    *,
    dataset: str,
    backend: Literal["torch", "jax", "keras"],
    mode: Literal["switch", "absolute"],
    residual: bool,
    epochs: int,
    metric: Literal["accuracy", "rmse", "mse"],
) -> BenchmarkConfig:
    """Sample a BenchmarkConfig for one (dataset, flavor) trial.

    `convex_fraction` is searched only for absolute mode; switch keeps 0.5.
    `activation` is fixed to "elu" in Phase 2a.

    :param trial: Optuna trial used to suggest hyperparameter values.
    :param dataset: Dataset name (used to name the config; no taxonomy logic here).
    :param backend: ML backend to target.
    :param mode: Monotonicity mode (`"absolute"` or `"switch"`).
    :param residual: Whether to use residual connections.
    :param epochs: Number of training epochs per trial.
    :param metric: Primary metric; propagated into `cfg.metrics` so the
        objective's metric and the training config always agree.
    :returns: A fully populated `BenchmarkConfig` ready for `run()`.
    """
    width = trial.suggest_categorical("width", [8, 16, 21, 32, 64])
    depth = trial.suggest_int("depth", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr_decay = trial.suggest_float("lr_decay", 0.85, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
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
