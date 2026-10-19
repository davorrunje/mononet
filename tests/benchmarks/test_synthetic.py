"""Tests for the synthetic monotone-regression generator (depth probe, #99).

The generator families must be genuinely *nonlinear* at high complexity to
be useful depth probes — ``test_high_c_families_are_nonlinear`` is the gate
that would have caught the original degeneracy (all families near-linear,
``teacher_relu`` byte-identical to ``teacher_elu`` on the pre-fix ``[0,1]^d``
domain).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from benchmarks.datasets.registry import load
from benchmarks.datasets.synthetic import _target_fn, synth_monotone

if TYPE_CHECKING:
    from pathlib import Path

_FAMILIES = ("additive", "teacher_relu", "teacher_elu", "lattice")


@pytest.mark.parametrize("kind", _FAMILIES)
def test_high_c_families_are_nonlinear(kind: str) -> None:
    # Non-degeneracy gate: a high-complexity target must NOT be linearly
    # fittable. If a family regressed to near-linear (as the pre-fix ones
    # did), R^2 -> 1.0 and this fails.
    b = synth_monotone(kind, 4, d=6, n_train=16000, n_test=4000, seed=7)  # type: ignore[arg-type]
    r2 = LinearRegression().fit(b.X_train, b.y_train).score(b.X_train, b.y_train)
    assert r2 < 0.7, f"{kind} high-c is near-linear (R^2={r2:.3f})"


def test_teacher_relu_and_elu_differ_at_high_c() -> None:
    # On the centered [-1,1]^d domain the strongly-negative-biased
    # preactivations straddle zero, so relu (hard clip at 0) and elu (smooth
    # floor at -1) produce genuinely different on-distribution targets, at
    # the "high" complexity level used by the registry (c=4).
    r = synth_monotone("teacher_relu", 4, d=6, n_train=4000, n_test=64, seed=0)
    e = synth_monotone("teacher_elu", 4, d=6, n_train=4000, n_test=64, seed=0)
    assert float(np.max(np.abs(r.y_train - e.y_train))) > 1e-3


@pytest.mark.parametrize("kind", _FAMILIES)
def test_targets_are_monotone_increasing_in_every_feature(kind: str) -> None:
    # Bump each feature up one at a time (holding the others fixed) and
    # assert the (seeded) target function never decreases -- the property
    # every family relies on non-negative weights + a monotone activation
    # (or nested max/min of monotone terms) to guarantee "by construction",
    # checked here numerically at the "high" complexity level.
    d = 6
    f = _target_fn(kind, c=4, d=d, seed=3)  # type: ignore[arg-type]
    rng = np.random.default_rng(11)
    x = rng.uniform(-1.0, 1.0, size=(200, d))
    base = f(x)
    for j in range(d):
        xp = x.copy()
        xp[:, j] += 0.3
        assert np.all(f(xp) - base >= -1e-8), f"{kind} not monotone in dim {j}"


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


def test_registry_load_high_complexity_bundle(tmp_path: Path) -> None:
    # `registry.load` on a registered "_chigh" (c=4) key returns the
    # generator-backed bundle at the pinned size (d=6, n_train=32000
    # (>20k so it draws the large-batch band), n_test=4000), ignoring
    # `data_dir` entirely (nonexistent tmp_path).
    b = load("synth_teacher_relu_chigh", data_dir=tmp_path)
    assert b.task == "regression"
    assert b.X_train.shape == (32000, 6)
    assert b.X_test.shape == (4000, 6)
    assert b.mono_increasing == tuple(range(6))
    assert b.mono_decreasing == ()
