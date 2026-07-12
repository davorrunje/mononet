# tests/benchmarks/test_synthetic.py
from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from benchmarks.datasets.synthetic import synth_monotone


@pytest.mark.parametrize("kind", ["additive", "teacher_relu", "teacher_elu", "lattice"])
def test_synth_is_monotone_and_shaped(
    kind: Literal["additive", "teacher_relu", "teacher_elu", "lattice"],
) -> None:
    b = synth_monotone(kind, c=4, d=6, n_train=500, n_test=200, seed=0)
    assert b.task == "regression"
    assert b.X_train.shape == (500, 6)
    assert b.X_test.shape == (200, 6)
    assert b.mono_increasing == (0, 1, 2, 3, 4, 5)
    assert b.mono_decreasing == ()
    assert np.isfinite(b.y_train).all()
    # numerical monotonicity: raising any single feature never lowers f
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(64, 6))
    from benchmarks.datasets.synthetic import _target_fn

    f = _target_fn(kind, c=4, d=6, seed=0)
    base = f(x)
    for j in range(6):
        xp = x.copy()
        xp[:, j] = np.minimum(1.0, x[:, j] + 0.1)
        assert (f(xp) - base >= -1e-9).all(), f"non-monotone in dim {j}"


def test_synth_deterministic_per_seed() -> None:
    a = synth_monotone("teacher_relu", c=4, seed=0)
    b = synth_monotone("teacher_relu", c=4, seed=0)
    assert np.array_equal(a.y_train, b.y_train)
    assert not np.array_equal(
        a.y_train, synth_monotone("teacher_relu", c=4, seed=1).y_train
    )
