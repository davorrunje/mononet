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
    :param mode: Monotonicity mode ("split" or "mixed").
    :param residual: Whether residual connections were used.
    :param seed: Random seed for this run.
    :param scores: Dict mapping metric names to scalar values.
    :param epochs_run: Number of training epochs completed (epochs-to-best when
        early stopping is active, otherwise the full ``cfg.epochs``).
    :param diverged: Whether the run diverged — its final loss was non-finite or
        exceeded 10x the predict-the-mean baseline (see
        :func:`benchmarks._common.runner.is_diverged`).
    """

    dataset: str
    backend: str
    mode: str
    residual: bool
    seed: int
    scores: dict[str, float]
    epochs_run: int
    diverged: bool = False


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


def bootstrap_delta(
    deep_values: np.ndarray,  # type: ignore[type-arg]
    shallow_values: np.ndarray,  # type: ignore[type-arg]
    *,
    lower_is_better: bool = False,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Seed-bootstrap the signed IQM Δ between a "deep" and a "shallow" arm.

    Independently resamples ``deep_values`` and ``shallow_values`` with
    replacement (``n_boot`` draws) and takes the IQM difference on each draw,
    sign-normalized so that positive always means the ``deep`` arm is better
    (see :func:`benchmarks._common.stage2_gate._signed_improvement` for the
    same convention applied to point estimates). Shared by the loan size-ladder
    report and the Stage-2 deep-vs-shallow gate so both use one bootstrap
    implementation.

    :param deep_values: Per-seed metric values for the "deep" arm.
    :param shallow_values: Per-seed metric values for the "shallow" arm.
    :param lower_is_better: Metric direction; sign-flips Δ so positive always
        means "deep is better".
    :param n_boot: Number of bootstrap resamples.
    :param seed: RNG seed for reproducibility.
    :returns: ``(delta_point, delta_lo, delta_hi)`` — the un-resampled point
        estimate and the 2.5/97.5 percentiles of the bootstrap distribution.
    """
    dv = np.asarray(deep_values, dtype=np.float64)
    sv = np.asarray(shallow_values, dtype=np.float64)
    sign = -1.0 if lower_is_better else 1.0
    point = sign * (interquartile_mean(dv) - interquartile_mean(sv))
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            sign
            * (
                interquartile_mean(rng.choice(dv, len(dv), replace=True))
                - interquartile_mean(rng.choice(sv, len(sv), replace=True))
            )
            for _ in range(n_boot)
        ]
    )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


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
