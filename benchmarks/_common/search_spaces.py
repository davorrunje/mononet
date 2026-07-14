"""Per-flavor Optuna search space producing a BenchmarkConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

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
    mode: Literal["split", "mixed", "alternate"],
    residual: bool,
    epochs: int,
    metric: Literal["accuracy", "rmse", "mse", "roc_auc"],
    n_train: int,
    deep: bool = False,
    search_activation: bool = False,
    max_depth: int = 4,
    embed_layers: int = 1,
) -> BenchmarkConfig:
    """Sample a BenchmarkConfig for one (dataset, flavor) trial.

    `convex_fraction` is searched only for mixed mode; split and alternate
    keep it fixed at 0.5. `mode="alternate"` always uses the composition-aware
    initialisation arm (``alt_init="composition"``); other modes leave
    ``alt_init`` unset (`None`). `activation` is searched over
    `{"relu", "elu", "softplus", "selu"}` only when `search_activation` is
    `True`; otherwise it is fixed to `"elu"`.

    :param trial: Optuna trial used to suggest hyperparameter values.
    :param dataset: Dataset name (labels the config).
    :param backend: ML backend to target.
    :param mode: Monotonicity mode (`"mixed"`, `"split"`, or `"alternate"`).
    :param residual: Whether to use residual connections.
    :param epochs: Number of training epochs per trial.
    :param metric: Primary metric; propagated into `cfg.metrics` so the
        objective's metric and the training config always agree. When
        `"roc_auc"`, `cfg.metrics` also includes `"accuracy"` so it is still
        reported alongside the primary metric.
    :param n_train: Number of rows in the training set; selects the ``batch_size``
        band (large-batch band if ``n_train >= _LARGE_BATCH_THRESHOLD``).
    :param deep: When ``True``, draw ``depth`` from the deep categorical band
        ``{6, 10, 16}`` (residual skips make these trainable); otherwise draw
        ``depth`` from the shallow range ``[1, max_depth]``. Only affects the
        ``depth`` dimension; all other hyperparameters are sampled identically.
    :param search_activation: When ``True``, sample ``activation`` from
        ``{"relu", "elu", "softplus", "selu"}``; otherwise fix it to ``"elu"``.
    :param max_depth: Upper bound of the shallow ``depth`` range (``[1,
        max_depth]``) used when ``deep`` is `False`.
    :param embed_layers: Number of non-monotone `Dense` layers in
        ``cfg.embed_hidden``, each sized ``width``.
    :returns: A fully populated `BenchmarkConfig` ready for `run()`.
    """
    width = trial.suggest_categorical("width", [8, 16, 21, 32, 64])
    if deep:
        depth = trial.suggest_categorical("depth", [6, 10, 16])
    else:
        depth = trial.suggest_int("depth", 1, max_depth)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr_decay = trial.suggest_float("lr_decay", 0.85, 1.0)
    batch_choices = (
        _BATCH_SIZES_LARGE if n_train >= _LARGE_BATCH_THRESHOLD else _BATCH_SIZES_SMALL
    )
    batch_size = trial.suggest_categorical("batch_size", batch_choices)
    activation = cast(
        "Literal['relu', 'elu', 'selu', 'softplus', 'identity']",
        (
            trial.suggest_categorical("activation", ["relu", "elu", "softplus", "selu"])
            if search_activation
            else "elu"
        ),
    )
    convex_fraction = (
        trial.suggest_float("convex_fraction", 0.0, 1.0) if mode == "mixed" else 0.5
    )
    alt_init: Literal["composition", "legacy"] | None = (
        "composition" if mode == "alternate" else None
    )
    embed_hidden = tuple(int(width) for _ in range(embed_layers))
    metrics: tuple[Literal["accuracy", "rmse", "mse", "roc_auc"], ...] = (
        ("roc_auc", "accuracy") if metric == "roc_auc" else (metric,)
    )
    return BenchmarkConfig(
        dataset=dataset,
        backend=backend,
        mode=mode,
        residual=residual,
        depth=depth,
        width=int(width),
        activation=activation,
        convex_fraction=convex_fraction,
        embed_hidden=embed_hidden,
        dropout=dropout,
        optimizer=OptimizerSpec("adam", lr, weight_decay),
        lr_decay=lr_decay,
        batch_size=int(batch_size),
        epochs=epochs,
        early_stopping=None,
        seeds=(0,),
        metrics=metrics,
        alt_init=alt_init,
    )
