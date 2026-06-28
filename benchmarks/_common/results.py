"""Per-run result records and best-k-of-n aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResultRow:
    """A single benchmark run result.

    :param dataset: Dataset name.
    :param backend: Backend used ("torch", "jax", or "keras").
    :param mode: Monotonicity mode ("switch" or "absolute").
    :param residual: Whether residual connections were used.
    :param seed: Random seed for this run.
    :param scores: Dict mapping metric names to scalar values.
    :param epochs_run: Number of training epochs completed.
    """

    dataset: str
    backend: str
    mode: str
    residual: bool
    seed: int
    scores: dict[str, float]
    epochs_run: int


@dataclass(frozen=True, slots=True)
class Aggregate:
    """Aggregated statistics over a selected subset of result rows.

    :param metric: Name of the metric being aggregated.
    :param mean: Mean value of the metric over selected rows.
    :param std: Standard deviation of the metric over selected rows.
    :param n_seeds: Total number of rows considered.
    :param n_selected: Number of rows selected (top-k).
    """

    metric: str
    mean: float
    std: float
    n_seeds: int
    n_selected: int


def aggregate(
    rows: list[ResultRow], *, metric: str, lower_is_better: bool, top_k: int = 5
) -> Aggregate:
    """Mean/std of the best `top_k` rows by `metric`.

    :param rows: List of ResultRow objects to aggregate.
    :param metric: Metric name to extract from each row's scores dict.
    :param lower_is_better: If True, smaller values are better; if False, larger values are better.
    :param top_k: Number of best rows to select (default 5).
    :returns: Aggregate object with mean, std, and counts.
    """
    vals = np.array([r.scores[metric] for r in rows], dtype=np.float64)
    order = np.argsort(vals)
    selected = order[:top_k] if lower_is_better else order[::-1][:top_k]
    best = vals[selected]
    return Aggregate(
        metric=metric,
        mean=float(best.mean()),
        std=float(best.std()),
        n_seeds=len(rows),
        n_selected=int(best.size),
    )


def write_jsonl(rows: list[ResultRow], path: Path) -> None:
    """Write result rows to a JSONL file.

    :param rows: List of ResultRow objects to serialize.
    :param path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
