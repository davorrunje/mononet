# SPDX-License-Identifier: Apache-2.0
"""Tests for mononet.core.numerics tolerance helpers."""

from __future__ import annotations

import numpy as np

from mononet.core.numerics import (
    ATOL_FLOAT32,
    ATOL_FLOAT64,
    RTOL_FLOAT32,
    RTOL_FLOAT64,
    default_atol,
    default_rtol,
)


def test_default_tolerances_for_float64() -> None:
    assert default_atol(np.float64) == ATOL_FLOAT64
    assert default_rtol(np.float64) == RTOL_FLOAT64


def test_default_tolerances_for_float32() -> None:
    assert default_atol(np.float32) == ATOL_FLOAT32
    assert default_rtol(np.float32) == RTOL_FLOAT32


def test_non_float64_dtype_falls_back_to_float32_tolerances() -> None:
    # any dtype that is not float64 uses the float32 tolerances
    assert default_atol(np.float16) == ATOL_FLOAT32
    assert default_rtol(np.float16) == RTOL_FLOAT32
