# SPDX-License-Identifier: Apache-2.0
"""Tests for monotonic_residual reference."""

from __future__ import annotations

import numpy as np

from mononet.core import reference as ref
from mononet.core.types import ActivationSpec


def test_warm_start_is_near_identity() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3)).astype(np.float64)
    w = rng.normal(size=(3, 3)).astype(np.float64)
    b = np.zeros(3)
    y = ref.monotonic_residual(
        x,
        w,
        b,
        np.array(0.0),
        np.array(0.0),
        mode="switch",
        activation=ActivationSpec("relu"),
    )
    # alpha gate = 1, beta gate = eps ~= 0 -> y ~= identity skip
    np.testing.assert_allclose(y, x, atol=5e-3)


def test_projection_when_dims_differ() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 2)).astype(np.float64)
    w = rng.normal(size=(2, 5)).astype(np.float64)
    b = np.zeros(5)
    skip = rng.normal(size=(2, 5)).astype(np.float64)
    y = ref.monotonic_residual(
        x,
        w,
        b,
        np.array(0.0),
        np.array(0.0),
        mode="switch",
        activation=ActivationSpec("relu"),
        skip_weight=skip,
    )
    assert y.shape == (4, 5)


def test_residual_is_nondecreasing() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 3)).astype(np.float64)
    w = rng.normal(size=(3, 3)).astype(np.float64)
    b = rng.normal(size=3).astype(np.float64)
    bump = np.zeros_like(x)
    bump[:, 0] = 0.4
    act = ActivationSpec("relu")
    y0 = ref.monotonic_residual(
        x, w, b, np.array(0.3), np.array(0.5), mode="switch", activation=act
    )
    y1 = ref.monotonic_residual(
        x + bump, w, b, np.array(0.3), np.array(0.5), mode="switch", activation=act
    )
    assert np.all(y1 - y0 >= -1e-9)
