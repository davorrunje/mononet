import math

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from mononet.core.init import absolute_init_params
from mononet.jax import MonoLinear


def test_absolute_default_weight_scale_and_bias() -> None:
    in_f, units = 256, 512
    layer = MonoLinear(in_f, units, mode="mixed", activation="elu", rngs=nnx.Rngs(0))
    gain, bias = absolute_init_params("elu", 0.5)
    got = float(jnp.std(layer.weight[...]))
    assert abs(got - gain / math.sqrt(in_f)) < 0.05 * gain / math.sqrt(in_f)
    assert layer.bias is not None
    assert jnp.allclose(layer.bias[...], jnp.full((units,), bias))


def test_absolute_bias_nonzero_off_half() -> None:
    layer = MonoLinear(
        64,
        64,
        mode="mixed",
        activation="elu",
        convex_fraction=0.25,
        rngs=nnx.Rngs(0),
    )
    _, bias = absolute_init_params("elu", 0.25)
    assert bias != 0.0
    assert layer.bias is not None
    assert jnp.allclose(layer.bias[...], jnp.full((64,), bias))


def test_switch_default_unchanged() -> None:
    layer = MonoLinear(64, 64, mode="split", activation="elu", rngs=nnx.Rngs(0))
    assert layer.bias is not None
    assert jnp.allclose(layer.bias[...], jnp.zeros((64,)))
