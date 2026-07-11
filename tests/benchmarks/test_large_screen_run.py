from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks.large_screen_run import screen_dataset


def _toy_bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + x[:, 1] - x[:, 2] > 0).astype(float)
    xt = rng.normal(size=(80, 3))
    yt = (xt[:, 0] + xt[:, 1] - xt[:, 2] > 0).astype(float)
    return DatasetBundle(
        name="toy",
        task="binary_classification",
        X_train=x,
        y_train=y,
        X_test=xt,
        y_test=yt,
        mono_increasing=(0, 1),
        mono_decreasing=(2,),
        feature_names=("f0", "f1", "f2"),
        metadata={},
    )


def test_screen_dataset_smoke() -> None:
    """Tiny budget end-to-end: a record with finite Δ and a valid verdict."""
    rec = screen_dataset(
        _toy_bundle(), n_trials=1, search_seeds=1, final_seeds=2, epochs=1, n_jobs=1
    )
    assert rec["name"] == "toy"
    assert np.isfinite(rec["delta"])
    assert rec["verdict"] in {"ladder", "standard"}
    assert rec["delta_lo"] <= rec["delta"] <= rec["delta_hi"]
