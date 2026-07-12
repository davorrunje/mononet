from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.monotone_depth_probe_run import probe_dataset


def test_probe_dataset_smoke() -> None:
    rec = probe_dataset(
        "additive", c=1, n_trials=1, search_seeds=1, final_seeds=2, epochs=1
    )
    assert rec["kind"] == "additive"
    assert rec["c"] == 1
    assert np.isfinite(rec["deep_mse_iqm"])
    assert np.isfinite(rec["shallow_mse_iqm"])
    assert len(rec["deep_values"]) == 2
    assert len(rec["shallow_values"]) == 2
