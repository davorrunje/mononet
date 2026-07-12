# SPDX-License-Identifier: Apache-2.0
"""Tests for the JAX (Flax NNX) model builders (monotonicity by construction)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp

from applications.pinn.core.problems import conservation
from applications.pinn.models.jax import builders
from applications.pinn.models.protocol import Method, ModelConfig

CFG = ModelConfig(width=16, n_blocks=2, t_embed_dim=4, seed=0)
METHODS: list[Method] = ["vanilla", "soft", "weight_clip", "hard_monotone"]


def _grid(n: int = 80) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = jnp.linspace(-1.5, 2.0, n).reshape(-1, 1)
    t = jnp.full((n, 1), 0.3)
    return x, t


@pytest.mark.parametrize("method", METHODS)
def test_forward_shape(method: Method) -> None:
    """Every method returns a column vector of the right shape."""
    model = builders.build_jax(conservation.BurgersRiemann(), CFG, method)
    x, t = _grid()
    assert model(x, t).shape == (80, 1)


def test_hard_monotone_decreasing_in_x() -> None:
    """A decreasing problem (u_l>u_r) yields a non-increasing field in x."""
    model = builders.build_jax(
        conservation.BurgersRiemann(u_l=1.0, u_r=0.0), CFG, "hard_monotone"
    )
    x, t = _grid()
    u = np.asarray(model(x, t)).ravel()
    assert np.all(np.diff(u) <= 1e-4)


def test_hard_monotone_increasing_in_x() -> None:
    """A forming-queue LWR problem (rho_l<rho_r) yields a non-decreasing field."""
    model = builders.build_jax(
        conservation.LwrRiemann(rho_l=0.2, rho_r=0.8), CFG, "hard_monotone"
    )
    x, t = _grid()
    u = np.asarray(model(x, t)).ravel()
    assert np.all(np.diff(u) >= -1e-4)


def test_weight_clip_is_monotone_in_x() -> None:
    """The weight-clip baseline is also monotone in x by construction."""
    model = builders.build_jax(
        conservation.BurgersRiemann(u_l=1.0, u_r=0.0), CFG, "weight_clip"
    )
    x, t = _grid()
    u = np.asarray(model(x, t)).ravel()
    assert np.all(np.diff(u) <= 1e-4)


def test_hard_monotone_depends_on_t() -> None:
    """The field genuinely varies with t (free, not constant in t)."""
    model = builders.build_jax(conservation.BurgersRiemann(), CFG, "hard_monotone")
    x = jnp.linspace(-1.0, 1.0, 40).reshape(-1, 1)
    u0 = np.asarray(model(x, jnp.zeros_like(x)))
    u1 = np.asarray(model(x, jnp.ones_like(x)))
    assert float(np.abs(u0 - u1).max()) > 1e-4
