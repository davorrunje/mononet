# SPDX-License-Identifier: Apache-2.0
"""Tests for scoring metrics and plotting."""

from __future__ import annotations

import numpy as np

from applications.pinn.core import metrics


def test_l1_l2() -> None:
    """L1/L2 errors match closed-form values."""
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 0.0, 0.0])
    assert metrics.l1(a, b) == 3.0
    assert np.isclose(metrics.l2(a, b), np.sqrt(5.0))
    assert metrics.l1(a, b, dx=0.5) == 1.5


def test_tv_of_monotone_is_endpoint_difference() -> None:
    """TV of a monotone profile equals |first - last|."""
    prof = np.array([3.0, 2.0, 1.0, 0.5])
    assert np.isclose(metrics.tv(prof), 2.5)


def test_tv_of_bump_exceeds_monotone() -> None:
    """A non-monotone profile has larger TV than its monotone envelope."""
    bump = np.array([3.0, 2.0, 2.6, 1.0])
    assert metrics.tv(bump) > 2.0


def test_tv_curve_shape_and_values() -> None:
    """tv_curve returns one TV per time slice."""
    field = np.array([[3.0, 2.0, 1.0], [2.0, 1.0, 0.0]])
    curve = metrics.tv_curve(field)
    assert curve.shape == (2,)
    assert np.allclose(curve, [2.0, 2.0])


def test_overshoot_zero_when_within_range() -> None:
    """A profile inside the reference range has zero overshoot."""
    ref = np.array([1.0, 0.5, 0.0])
    pred = np.array([0.9, 0.4, 0.1])
    assert metrics.overshoot(pred, ref) == 0.0


def test_overshoot_measures_excursion() -> None:
    """Overshoot equals the largest excursion beyond the reference range."""
    ref = np.array([1.0, 0.0])
    pred = np.array([1.2, -0.1])  # 0.2 above, 0.1 below -> 0.2
    assert np.isclose(metrics.overshoot(pred, ref), 0.2)


def test_shock_position_error() -> None:
    """Front-position error tracks a shifted level crossing."""
    x = np.linspace(0.0, 1.0, 101)
    ref = 1.0 - x  # crosses 0.5 at x = 0.5
    pred = 1.1 - x  # crosses 0.5 at x = 0.6
    err = metrics.shock_position_error(pred, ref, x, level=0.5)
    assert np.isclose(err, 0.1, atol=0.02)


def test_mass_error() -> None:
    """Mass error is the absolute difference of integrals."""
    a = np.array([1.0, 1.0, 1.0])
    b = np.array([1.0, 0.0, 1.0])
    assert np.isclose(metrics.mass_error(a, b, dx=0.5), 0.5)


def test_reconstruction_error_over_field() -> None:
    """Reconstruction error is the L2 over the flattened field."""
    field = np.zeros((2, 3))
    ref = np.ones((2, 3))
    assert np.isclose(metrics.reconstruction_error(field, ref), np.sqrt(6.0))
