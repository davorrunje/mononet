"""Tests for the synthetic monotone-regression generator (depth probe, #99).

The generator families must be genuinely *nonlinear* at high complexity to
be useful depth probes — the ``test_high_c_families_are_nonlinear`` gate is
the check that would have caught the original degeneracy (all families
near-linear, ``teacher_relu`` byte-identical to ``teacher_elu``).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from benchmarks.datasets.synthetic import _target_fn, synth_monotone

_FAMILIES = ("additive", "teacher_relu", "teacher_elu", "lattice")


def test_teacher_relu_and_elu_differ() -> None:
    # On the centered [-1,1]^d domain the strongly-negative-biased
    # preactivations straddle zero, so relu (hard clip at 0) and elu (smooth
    # floor at -1) produce genuinely different on-distribution targets. This
    # guards the activation switch against the original degeneracy where they
    # were bit-identical.
    r = synth_monotone("teacher_relu", 2, d=6, n_train=4000, n_test=64, seed=0)
    e = synth_monotone("teacher_elu", 2, d=6, n_train=4000, n_test=64, seed=0)
    assert float(np.max(np.abs(r.y_train - e.y_train))) > 1e-3


@pytest.mark.parametrize("kind", _FAMILIES)
def test_high_c_families_are_nonlinear(kind: str) -> None:
    # Non-degeneracy gate: a high-complexity target must NOT be linearly
    # fittable. If a family regressed to near-linear (as the pre-fix ones
    # did), R^2 -> 1.0 and this fails.
    b = synth_monotone(kind, 4, d=6, n_train=16000, n_test=4000, seed=7)  # type: ignore[arg-type]
    r2 = LinearRegression().fit(b.X_train, b.y_train).score(b.X_train, b.y_train)
    assert r2 < 0.7, f"{kind} high-c is near-linear (R^2={r2:.3f})"


def test_targets_are_monotone_increasing() -> None:
    b = synth_monotone("teacher_relu", 2, d=6, n_train=512, n_test=64, seed=0)
    f = _target_fn("teacher_relu", c=2, d=6, seed=0)
    x = b.X_train.copy()
    base = f(x)
    x[:, 0] += 0.5  # raise one increasing feature
    assert np.all(f(x) - base >= -1e-8)


def test_deterministic() -> None:
    a = synth_monotone("lattice", 2, d=6, n_train=128, n_test=32, seed=1)
    b = synth_monotone("lattice", 2, d=6, n_train=128, n_test=32, seed=1)
    assert np.array_equal(a.y_train, b.y_train)
    assert np.array_equal(a.X_train, b.X_train)


def test_bundle_shape_and_monotone_direction() -> None:
    b = synth_monotone("additive", 1, d=6, n_train=100, n_test=20, seed=2)
    assert b.task == "regression"
    assert b.X_train.shape == (100, 6)
    assert b.X_test.shape == (20, 6)
    assert b.mono_increasing == tuple(range(6))
    assert b.mono_decreasing == ()
