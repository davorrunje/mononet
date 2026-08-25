# SPDX-License-Identifier: Apache-2.0
"""Error and admissibility metrics for scoring PINN solutions.

All metrics operate on NumPy arrays sampled on the evaluation grid and are used
identically across backends and methods.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]


def l1(pred: Array, ref: Array, *, dx: float = 1.0) -> float:
    """Discrete L1 error ``sum|pred - ref| * dx``."""
    return float(np.abs(pred - ref).sum() * dx)


def l2(pred: Array, ref: Array, *, dx: float = 1.0) -> float:
    """Discrete L2 error ``sqrt(sum(pred - ref)^2 * dx)``."""
    return float(np.sqrt((np.square(pred - ref)).sum() * dx))


def tv(profile: Array) -> float:
    """Total variation of a 1-D profile, ``sum|diff|``."""
    return float(np.abs(np.diff(profile)).sum())


def tv_curve(field: Array) -> Array:
    """Total variation at each time slice of a ``(nt, nx)`` field."""
    per_slice: Array = np.abs(np.diff(field, axis=1)).sum(axis=1)
    return per_slice


def overshoot(pred: Array, ref: Array) -> float:
    """Excursion of ``pred`` beyond the range of ``ref`` (the Gibbs overshoot).

    Zero when ``pred`` stays within ``[min(ref), max(ref)]``; otherwise the
    largest amount by which it exceeds that range on either side.

    :param pred: Predicted values.
    :param ref: Reference values.
    :returns: Non-negative overshoot magnitude.
    """
    above = float(np.max(pred) - np.max(ref))
    below = float(np.min(ref) - np.min(pred))
    return max(above, below, 0.0)


def shock_position(profile: Array, x_values: Array, level: float) -> float:
    """Locate the ``level`` crossing of a monotone profile (the front position).

    :param profile: 1-D solution values on ``x_values``.
    :param x_values: Spatial grid.
    :param level: Crossing level (e.g. the mean of the two states).
    :returns: The x-coordinate nearest the level crossing.
    """
    idx = int(np.argmin(np.abs(profile - level)))
    return float(x_values[idx])


def shock_position_error(
    pred: Array, ref: Array, x_values: Array, *, level: float
) -> float:
    """Absolute error in the front position between ``pred`` and ``ref``."""
    return abs(
        shock_position(pred, x_values, level) - shock_position(ref, x_values, level)
    )


def mass_error(pred: Array, ref: Array, *, dx: float = 1.0) -> float:
    """Absolute difference in total mass ``|sum(pred) - sum(ref)| * dx``."""
    return float(abs(pred.sum() - ref.sum()) * dx)


def reconstruction_error(field: Array, ref: Array, *, dx: float = 1.0) -> float:
    """L2 error over a whole ``(nt, nx)`` field (inverse-flagship headline)."""
    return l2(field.ravel(), ref.ravel(), dx=dx)
