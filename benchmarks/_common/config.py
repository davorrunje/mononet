"""Benchmark run configuration (sweep-ready value objects)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    """Optimizer configuration.

    :param name: Optimizer name (currently only "adam" is supported).
    :param lr: Learning rate.
    :param weight_decay: L2 regularization coefficient.
    """

    name: Literal["adam"]
    lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True, slots=True)
class EarlyStoppingSpec:
    """Early stopping configuration.

    :param monitor: Metric to monitor (e.g., "val_mse").
    :param patience: Number of epochs without improvement before stopping.
    """

    monitor: str
    patience: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Complete benchmark run configuration.

    Frozen dataclass with `.replace(**kw)` for sweep-ready immutable configuration.

    :param dataset: Dataset name.
    :param backend: Target backend ("torch", "jax", or "keras").
    :param mode: Monotonicity mode ("switch" or "absolute").
    :param residual: Whether to use residual connections.
    :param depth: Network depth (number of layers).
    :param width: Network width (hidden units per layer).
    :param activation: Activation function name (e.g., "elu").
    :param convex_fraction: Fraction of outputs constrained to convexity.
    :param embed_hidden: Tuple of embedding layer widths.
    :param dropout: Dropout rate.
    :param optimizer: OptimizerSpec instance.
    :param lr_decay: Optional learning rate decay coefficient.
    :param batch_size: Training batch size.
    :param epochs: Number of training epochs.
    :param early_stopping: Optional EarlyStoppingSpec instance.
    :param seeds: Tuple of random seeds to use.
    :param metrics: Tuple of metric names to track.
    """

    dataset: str
    backend: Literal["torch", "jax", "keras"]
    mode: Literal["switch", "absolute"]
    residual: bool
    depth: int
    width: int
    activation: str
    convex_fraction: float
    embed_hidden: tuple[int, ...]
    dropout: float
    optimizer: OptimizerSpec
    lr_decay: float | None
    batch_size: int
    epochs: int
    early_stopping: EarlyStoppingSpec | None
    seeds: tuple[int, ...]
    metrics: tuple[Literal["accuracy", "rmse", "mse"], ...]

    def replace(self, **changes: Any) -> BenchmarkConfig:
        """Return a copy with the given fields overridden.

        :param changes: Field names and values to override.
        :returns: New BenchmarkConfig with specified changes applied.
        """
        return dataclasses.replace(self, **changes)
