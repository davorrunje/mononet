"""Stage-2 deep-vs-shallow gate: signed Δ + seed-bootstrap CI + verdict.

Given one dataset's six Stage-A flavor result JSONs
(``{split,mixed} x {plain,residual,deep}``), :func:`dataset_delta` picks
the best shallow flavor (of the four non-deep) and the best deep flavor (of
the two deep), and seed-bootstraps a signed Δ (positive == deep is better,
sign-normalized by metric direction) with a 95% CI. :func:`verdict` then
classifies that Δ given a practical-significance margin chosen *after*
looking at the raw numbers — the margin is never hardcoded here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from benchmarks._common.results import bootstrap_delta
from benchmarks._common.screen_gate import gate
from benchmarks._common.search import _lower_is_better

if TYPE_CHECKING:
    from pathlib import Path

_SHALLOW_FLAVORS: tuple[str, ...] = (
    "split-plain",
    "split-residual",
    "mixed-plain",
    "mixed-residual",
)
_DEEP_FLAVORS: tuple[str, ...] = ("split-deep", "mixed-deep")


@dataclass(frozen=True, slots=True)
class DeltaResult:
    """Signed deep-vs-shallow Δ with its 95% seed-bootstrap CI.

    :param delta_point: Point estimate of Δ (positive == deep better).
    :param delta_lo: Lower bound of the 95% seed-bootstrap band.
    :param delta_hi: Upper bound of the 95% seed-bootstrap band.
    :param best_shallow_flavor: Flavor label of the best of the 4 shallow arms.
    :param best_deep_flavor: Flavor label of the best of the 2 deep arms.
    """

    delta_point: float
    delta_lo: float
    delta_hi: float
    best_shallow_flavor: str
    best_deep_flavor: str


def _signed_improvement(*, deep: float, shallow: float, lower_is_better: bool) -> float:
    """Δ normalized so positive == deep is better, regardless of metric direction.

    :param deep: Deep arm's metric value.
    :param shallow: Shallow arm's metric value.
    :param lower_is_better: Metric direction.
    :returns: ``shallow - deep`` when lower is better, else ``deep - shallow``.
    """
    return (shallow - deep) if lower_is_better else (deep - shallow)


def verdict(
    delta: DeltaResult, *, margin: float
) -> Literal["deep-better", "neutral", "deep-worse"]:
    """Classify a dataset's deep-vs-shallow gap given a chosen practical margin.

    :param delta: Signed Δ (positive == deep better) with its 95% CI.
    :param margin: Practical-significance floor (chosen post-results).
    :returns: ``deep-better`` when significantly and practically positive;
        ``deep-worse`` when significantly negative; else ``neutral``.
    """
    if gate(delta.delta_lo, delta.delta_point, margin) == "ladder":
        return "deep-better"
    if delta.delta_hi < 0.0 and abs(delta.delta_point) >= margin:
        return "deep-worse"
    return "neutral"


def _load_flavor(result_dir: Path, dataset: str, flavor: str) -> dict[str, Any]:
    path = result_dir / f"{dataset}-{flavor}.json"
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def _iqm_key(record: dict[str, Any]) -> float:
    return float(record["test_iqm"])


def _best_flavor(
    records: list[dict[str, Any]], *, lower_is_better: bool
) -> dict[str, Any]:
    return min(records, key=_iqm_key) if lower_is_better else max(records, key=_iqm_key)


def dataset_delta(
    result_dir: Path,
    dataset: str,
    metric: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> DeltaResult:
    """Load one dataset's 6 flavor result JSONs and compute the deep-vs-shallow Δ.

    Selects the best of the 4 shallow flavors (``{split,mixed} x
    {plain,residual}``) and the best of the 2 deep flavors
    (``{split,mixed}-deep``) by their reported ``test_iqm``, then
    seed-bootstraps the signed Δ (see :func:`_signed_improvement`) between
    their per-seed ``test_values`` via
    :func:`benchmarks._common.results.bootstrap_delta`.

    :param result_dir: Directory containing the ``{dataset}-{flavor}.json``
        files written by :func:`benchmarks._common.search.run_dataset`.
    :param dataset: Dataset name (filename prefix).
    :param metric: Primary metric name; must match every flavor's
        ``test_metric``.
    :param n_boot: Number of seed-bootstrap resamples.
    :param seed: RNG seed for the bootstrap.
    :returns: The signed Δ (positive == deep better) with its 95% CI and the
        two selected flavor labels.
    :raises ValueError: If a loaded flavor's ``test_metric`` does not match
        ``metric``.
    """
    lower = _lower_is_better(metric)
    shallow_recs = [_load_flavor(result_dir, dataset, f) for f in _SHALLOW_FLAVORS]
    deep_recs = [_load_flavor(result_dir, dataset, f) for f in _DEEP_FLAVORS]
    for rec in (*shallow_recs, *deep_recs):
        if rec["test_metric"] != metric:
            raise ValueError(
                f"flavor {rec['flavor']!r} has test_metric "
                f"{rec['test_metric']!r}, expected {metric!r}"
            )

    best_shallow = _best_flavor(shallow_recs, lower_is_better=lower)
    best_deep = _best_flavor(deep_recs, lower_is_better=lower)

    delta_point, delta_lo, delta_hi = bootstrap_delta(
        best_deep["test_values"],
        best_shallow["test_values"],
        lower_is_better=lower,
        n_boot=n_boot,
        seed=seed,
    )
    return DeltaResult(
        delta_point=delta_point,
        delta_lo=delta_lo,
        delta_hi=delta_hi,
        best_shallow_flavor=best_shallow["flavor"],
        best_deep_flavor=best_deep["flavor"],
    )
