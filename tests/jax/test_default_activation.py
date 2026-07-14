import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from mononet.jax import MonoLinear


def test_default_activation_is_affine() -> None:
    layer = MonoLinear(4, 8, mode="split", rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    x1 = jnp.asarray(rng.standard_normal((5, 4)), dtype=jnp.float32)
    x2 = jnp.asarray(rng.standard_normal((5, 4)), dtype=jnp.float32)
    mid = layer((x1 + x2) / 2)
    avg = (layer(x1) + layer(x2)) / 2
    np.testing.assert_allclose(np.asarray(mid), np.asarray(avg), rtol=1e-5, atol=1e-5)
