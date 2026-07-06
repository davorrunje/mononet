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
    """Aggregated statistics over a set of result rows, under several protocols.

    Multiple estimators are reported side by side so the benchmark can be read
    under a paper-comparable protocol (``mean``/``std``) and under
    outlier-robust protocols (``median``, ``iqm``) that survive the occasional
    seed collapse (see [[stage2-collapse-investigation]]).

    :param metric: Name of the metric being aggregated.
    :param mean: Mean over all rows (paper-comparable protocol).
    :param std: Standard deviation over all rows.
    :param median: Median over all rows (robust to outliers).
    :param iqm: Interquartile mean — mean of the middle 50% after trimming the
        top and bottom 25% (rliable-style; robust to collapses *and* lucky runs).
    :param n_seeds: Total number of rows considered.
    :param n_selected: Number of rows selected (top-k; equals ``n_seeds`` when
        no selection is applied).
    :param values: Per-seed metric values, in the input row order.
    """

    metric: str
    mean: float
    std: float
    median: float
    iqm: float
    n_seeds: int
    n_selected: int
    values: tuple[float, ...]


def interquartile_mean(vals: np.ndarray) -> float:  # type: ignore[type-arg]
    """Mean of the middle 50% of ``vals`` (drop top and bottom 25%).

    Falls back to the plain mean when trimming would remove everything (very
    small samples). See Agarwal et al., *Deep RL at the Edge of the Statistical
    Precipice* (NeurIPS 2021) for the IQM rationale.

    :param vals: 1-D array of metric values.
    :returns: The interquartile mean as a float.
    """
    v = np.sort(np.asarray(vals, dtype=np.float64))
    n = v.size
    k = int(np.floor(n * 0.25))
    core = v[k : n - k] if n - 2 * k > 0 else v
    return float(core.mean())


def aggregate(
    rows: list[ResultRow], *, metric: str, lower_is_better: bool, top_k: int = 5
) -> Aggregate:
    """Multi-protocol statistics over ``rows`` for ``metric``.

    ``mean``/``std`` are taken over the best ``top_k`` rows (``top_k >= len(rows)``
    disables selection); the robust estimators (``median``, ``iqm``) and the
    stored per-seed ``values`` always cover **all** rows.

    :param rows: List of ResultRow objects to aggregate.
    :param metric: Metric name to extract from each row's scores dict.
    :param lower_is_better: If True, smaller values are better; if False, larger values are better.
    :param top_k: Number of best rows for the mean/std protocol (default 5).
    :returns: Aggregate object with mean/std/median/iqm and the per-seed values.
    """
    vals = np.array([r.scores[metric] for r in rows], dtype=np.float64)
    order = np.argsort(vals)
    selected = order[:top_k] if lower_is_better else order[::-1][:top_k]
    best = vals[selected]
    return Aggregate(
        metric=metric,
        mean=float(best.mean()),
        std=float(best.std()),
        median=float(np.median(vals)),
        iqm=interquartile_mean(vals),
        n_seeds=len(rows),
        n_selected=int(best.size),
        values=tuple(float(v) for v in vals),
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
