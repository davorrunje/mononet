from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import pytest

from benchmarks._common.stage2_gate import DeltaResult, dataset_delta, verdict

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("lo", "point", "margin", "expect"),
    [
        (0.01, 0.03, 0.02, "deep-better"),  # significant + clears margin
        (0.01, 0.015, 0.02, "neutral"),  # significant but below margin
        (-0.01, 0.03, 0.02, "neutral"),  # CI touches 0
        (-0.05, -0.03, 0.02, "deep-worse"),  # significantly worse
    ],
)
def test_verdict(lo: float, point: float, margin: float, expect: str) -> None:
    d = DeltaResult(
        delta_point=point,
        delta_lo=lo,
        delta_hi=point + 0.01,
        best_shallow_flavor="absolute-plain",
        best_deep_flavor="absolute-deep",
    )
    assert verdict(d, margin=margin) == expect


def test_delta_sign_is_normalized_for_lower_is_better() -> None:
    # deep MSE 0.10 vs shallow 0.15 -> deep BETTER -> positive delta
    from benchmarks._common.stage2_gate import _signed_improvement

    assert _signed_improvement(
        deep=0.10, shallow=0.15, lower_is_better=True
    ) == pytest.approx(0.05)
    # deep AUC 0.80 vs shallow 0.75 -> deep BETTER -> positive delta
    assert _signed_improvement(
        deep=0.80, shallow=0.75, lower_is_better=False
    ) == pytest.approx(0.05)


def _write_flavor(
    result_dir: Path,
    dataset: str,
    flavor: str,
    *,
    metric: str,
    values: list[float],
) -> None:
    import numpy as np

    from benchmarks._common.results import interquartile_mean

    arr = np.asarray(values, dtype=np.float64)
    rec: dict[str, Any] = {
        "dataset": dataset,
        "flavor": flavor,
        "best_params": {},
        "cv_best": 0.0,
        "test_metric": metric,
        "test_mean": float(arr.mean()),
        "test_std": float(arr.std()),
        "test_median": float(np.median(arr)),
        "test_iqm": interquartile_mean(arr),
        "test_values": values,
        "n_collapse": 0,
        "n_seeds": len(values),
    }
    (result_dir / f"{dataset}-{flavor}.json").write_text(json.dumps(rec))


def test_dataset_delta_smoke(tmp_path: Path) -> None:
    """Smoke-test dataset_delta on a tiny hand-written set of flavor JSONs.

    Returns a finite Δ with lo <= point <= hi.
    """
    dataset = "toy"
    metric = "accuracy"  # higher is better
    shallow_flavors = (
        "switch-plain",
        "switch-residual",
        "absolute-plain",
        "absolute-residual",
    )
    deep_flavors = ("switch-deep", "absolute-deep")
    for i, flavor in enumerate(shallow_flavors):
        _write_flavor(
            tmp_path, dataset, flavor, metric=metric, values=[0.50 + i * 0.01, 0.51]
        )
    for i, flavor in enumerate(deep_flavors):
        _write_flavor(
            tmp_path, dataset, flavor, metric=metric, values=[0.60 + i * 0.01, 0.61]
        )

    result = dataset_delta(tmp_path, dataset, metric, n_boot=200, seed=0)

    assert math.isfinite(result.delta_point)
    assert math.isfinite(result.delta_lo)
    assert math.isfinite(result.delta_hi)
    assert result.delta_lo <= result.delta_point <= result.delta_hi
    assert result.best_deep_flavor in deep_flavors
    assert result.best_shallow_flavor in shallow_flavors
