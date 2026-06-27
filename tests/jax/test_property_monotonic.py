# SPDX-License-Identifier: Apache-2.0
"""Property test: JAX layers are non-decreasing in every input."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from mononet.jax import MonoLinear  # noqa: E402


@settings(deadline=None, max_examples=40)
@given(mode=st.sampled_from(["switch", "absolute"]), seed=st.integers(0, 10_000))
def test_mono_linear_nondecreasing(mode: str, seed: int) -> None:
    layer = MonoLinear(3, 4, mode=mode, rngs=nnx.Rngs(seed))
    key = jax.random.key(seed)
    x = jax.random.normal(key, (6, 3))
    bump = jnp.zeros_like(x).at[:, 1].set(0.5)
    delta = layer(x + bump) - layer(x)
    assert bool(jnp.all(delta >= -1e-5))
