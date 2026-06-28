"""Benchmark configuration I/O (TOML loading and parsing)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec


def load_config(
    path: Path | str,
    *,
    backend: Literal["torch", "jax", "keras"],
    mode: Literal["switch", "absolute"],
    residual: bool,
) -> BenchmarkConfig:
    """Load benchmark configuration from a TOML file.

    :param path: Path to the TOML config file.
    :param backend: Target backend ("torch", "jax", or "keras").
    :param mode: Monotonicity mode ("switch" or "absolute").
    :param residual: Whether to use residual connections.
    :returns: Loaded BenchmarkConfig with backend/mode/residual set from arguments.
    """
    path = Path(path)

    with path.open("rb") as f:
        data: dict[str, Any] = tomllib.load(f)

    dataset_section = data.get("dataset", {})
    model_section = data.get("model", {})
    train_section = data.get("train", {})

    dataset_name: str = dataset_section.get("name", path.stem)
    depth: int = model_section.get("depth", 2)
    width: int = model_section.get("width", 16)
    activation: str = model_section.get("activation", "elu")
    convex_fraction: float = model_section.get("convex_fraction", 0.5)
    embed_hidden_list: list[int] = model_section.get("embed_hidden", [])

    optimizer_name: str = train_section.get("optimizer", "adam")
    lr: float = train_section.get("lr", 0.001)
    weight_decay: float = train_section.get("weight_decay", 0.0)
    dropout: float = train_section.get("dropout", 0.0)
    lr_decay: float | None = train_section.get("lr_decay", None)
    batch_size: int = train_section.get("batch_size", 32)
    epochs: int = train_section.get("epochs", 50)
    seeds_list: list[int] = train_section.get("seeds", [0])
    metrics_list: list[str] = train_section.get("metrics", [])

    optimizer = OptimizerSpec(
        name=optimizer_name,  # type: ignore
        lr=lr,
        weight_decay=weight_decay,
    )

    return BenchmarkConfig(
        dataset=dataset_name,
        backend=backend,
        mode=mode,
        residual=residual,
        depth=depth,
        width=width,
        activation=activation,
        convex_fraction=convex_fraction,
        embed_hidden=tuple(embed_hidden_list),
        dropout=dropout,
        optimizer=optimizer,
        lr_decay=lr_decay,
        batch_size=batch_size,
        epochs=epochs,
        early_stopping=None,
        seeds=tuple(seeds_list),
        metrics=tuple(metrics_list),  # type: ignore
    )
