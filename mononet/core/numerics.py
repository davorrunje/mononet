# SPDX-License-Identifier: Apache-2.0
"""Numerical tolerances and dtype helpers shared by all backends."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Default tolerances used by the cross-backend equivalence harness.
ATOL_FLOAT32 = 1e-5
RTOL_FLOAT32 = 1e-5
ATOL_FLOAT64 = 1e-6
RTOL_FLOAT64 = 1e-6


def default_atol(dtype: npt.DTypeLike) -> float:
    """Return the default absolute tolerance for a given floating dtype."""
    d = np.dtype(dtype)
    if d == np.float64:
        return ATOL_FLOAT64
    return ATOL_FLOAT32


def default_rtol(dtype: npt.DTypeLike) -> float:
    """Return the default relative tolerance for a given floating dtype."""
    d = np.dtype(dtype)
    if d == np.float64:
        return RTOL_FLOAT64
    return RTOL_FLOAT32
