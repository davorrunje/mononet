"""Contract test for the mononet.jax public API surface."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402


def test_exports() -> None:
    import mononet.jax as j

    assert set(j.__all__) == {"MonoLinear", "MonoResidual", "MonoInput"}


def test_mono_linear_runs() -> None:
    from mononet.jax import MonoLinear

    layer = MonoLinear(3, 5, mode="switch", rngs=nnx.Rngs(0))
    y = layer(jnp.zeros((2, 3)))
    assert y.shape == (2, 5)


def test_mono_residual_warm_start_near_identity() -> None:
    from mononet.jax import MonoResidual

    block = MonoResidual(4, 4, mode="switch", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (3, 4))
    assert jnp.allclose(block(x), x, atol=5e-3)


def test_mono_input_flips_signs() -> None:
    from mononet.core.types import MonotonicityMask
    from mononet.jax import MonoInput

    layer = MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(layer(x), jnp.array([[1.0, -2.0, 3.0]]))
