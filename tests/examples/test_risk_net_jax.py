from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from tests.examples._loader import load_example


def test_risk_net_forward_and_monotone() -> None:
    """JAX RiskNet runs and is monotone in x_mono (dirs +1, +1, -1)."""
    mod = load_example("risk_net_jax.py")
    net = mod.RiskNet(rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    x_mono = jnp.asarray(rng.standard_normal((16, 3)), dtype=jnp.float32)
    x_free = jnp.asarray(rng.standard_normal((16, 2)), dtype=jnp.float32)
    base = np.asarray(net(x_mono, x_free))
    assert base.shape == (16, 1)
    for j, sign in ((0, 1), (1, 1), (2, -1)):
        bumped = x_mono.at[:, j].add(0.5)
        diff = (np.asarray(net(bumped, x_free)) - base)[:, 0]
        assert (diff >= -1e-4).all() if sign > 0 else (diff <= 1e-4).all()
