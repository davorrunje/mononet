from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks.loan_size_ladder_run import run_ladder


def _synthetic_bundle(n: int = 600) -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 4))
    # monotone-ish label: increasing in col 0/1, decreasing in col 2
    logit = x[:, 0] + x[:, 1] - x[:, 2]
    y = (logit > 0).astype(np.float64)
    xt = rng.standard_normal((120, 4))
    yt = ((xt[:, 0] + xt[:, 1] - xt[:, 2]) > 0).astype(np.float64)
    return DatasetBundle(
        name="synthetic",
        task="binary_classification",
        X_train=x,
        y_train=y,
        X_test=xt,
        y_test=yt,
        mono_increasing=(0, 1),
        mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"),
        metadata={},
    )


def test_run_ladder_smoke() -> None:
    """run_ladder returns one finite-IQM record per (n, arm) on a tiny bundle."""
    recs = run_ladder(
        _synthetic_bundle(),
        ns=(100, 400),
        arms=("shallow", "deep"),
        n_trials=2,
        search_seeds=1,
        final_seeds=range(2),
        epochs=1,
    )
    assert len(recs) == 4  # 2 ns x 2 arms
    for r in recs:
        assert r["n"] in (100, 400)
        assert r["arm"] in ("shallow", "deep")
        assert np.isfinite(r["test_iqm"])
        assert len(r["test_values"]) == 2
