# SPDX-License-Identifier: Apache-2.0
"""Tests for the admissibility spec and violation metric."""

from __future__ import annotations

import numpy as np

from applications.pinn.core.admissibility import AdmissibilitySpec, violation


def test_monotone_decreasing_field_has_zero_violation() -> None:
    """A strictly decreasing profile violates a non-increasing target by 0."""
    x = np.linspace(0.0, 1.0, 50)
    field = 1.0 - x  # decreasing in x (axis 0)
    assert violation(field, axis=0, sign=-1) == 0.0


def test_bump_violation_equals_its_rise() -> None:
    """For a non-increasing target, violation = total upward jump (the rise)."""
    field = np.array([3.0, 2.0, 2.5, 1.0])  # one upward step of +0.5
    assert violation(field, axis=0, sign=-1) == 0.5


def test_increasing_target_sign_is_symmetric() -> None:
    """For a non-decreasing target, downward steps are the violations."""
    field = np.array([0.0, 1.0, 0.7, 2.0])  # one downward step of -0.3
    assert np.isclose(violation(field, axis=0, sign=1), 0.3)


def test_unconstrained_axis_never_violates() -> None:
    """A sign of 0 (unconstrained) yields zero violation regardless."""
    field = np.array([[0.0, 5.0], [9.0, -3.0]])
    assert violation(field, axis=1, sign=0) == 0.0


def test_violation_along_named_axis_of_2d_field() -> None:
    """Violation is measured along the requested axis of a 2-D field."""
    # rows decreasing in axis 0; columns flat -> zero violation on axis 0
    field = np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]])
    assert violation(field, axis=0, sign=-1) == 0.0


def test_spec_fields() -> None:
    """AdmissibilitySpec stores mask and convex/concave axes."""
    spec = AdmissibilitySpec(mask=(-1, 0), convex_axes=(), concave_axes=())
    assert spec.mask == (-1, 0)
    assert spec.convex_axes == ()
