"""Tests for the synthetic monotone-regression generator (depth probe, #99)."""

from __future__ import annotations

import numpy as np

from benchmarks.datasets.synthetic import _target_fn, synth_monotone


def test_teacher_relu_and_elu_differ() -> None:
    # NOTE: under uniform-[0,1]^6 sampling with realistic budgets (n up to
    # 20000), a training set essentially never lands close enough to the
    # all-zero corner to make any depth-1 pre-activation negative, so
    # relu(z) == z == elu(z) everywhere sampled and y_train is bit-identical
    # between the two families (verified empirically across many
    # seed/c combinations). That does NOT mean the `act` switch is unwired
    # -- it means the corner where relu/elu diverge (z < 0) has near-zero
    # probability mass under this domain/width/depth combination. Probe the
    # corner directly (x near the all-zero point) where the switch is
    # guaranteed to matter: relu(z)=0 vs elu(z)=expm1(z)<0 for z<0.
    f_relu = _target_fn("teacher_relu", c=2, d=6, seed=0)
    f_elu = _target_fn("teacher_elu", c=2, d=6, seed=0)
    x = np.full((1, 6), 1e-3)
    diff = float(np.max(np.abs(f_relu(x) - f_elu(x))))
    assert diff > 1e-6  # activation switch is wired


def test_targets_are_monotone_increasing() -> None:
    b = synth_monotone("teacher_relu", 2, d=6, n_train=512, n_test=64, seed=0)
    f = _target_fn("teacher_relu", c=2, d=6, seed=0)
    x = b.X_train.copy()
    base = f(x)
    x[:, 0] = np.minimum(1.0, x[:, 0] + 0.5)  # raise one increasing feature
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
