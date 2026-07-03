import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from mononet.jax import MonoLinear, MonoResidual


def test_default_builds_two_monolinears() -> None:
    layer = MonoResidual(8, 8, mode="absolute", activation="elu", rngs=nnx.Rngs(0))
    assert isinstance(layer.F, nnx.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F.layers) == 2


def test_subdepth_builds_k_monolinears() -> None:
    layer = MonoResidual(
        8, 8, mode="absolute", activation="elu", sub_depth=3, rngs=nnx.Rngs(0)
    )
    assert isinstance(layer.F, nnx.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F.layers) == 3


def test_subdepth1_is_single_monolinear() -> None:
    layer = MonoResidual(8, 8, mode="absolute", sub_depth=1, rngs=nnx.Rngs(0))
    assert isinstance(layer.F, MonoLinear)


def test_F_alone_is_used() -> None:  # noqa: N802
    f = MonoLinear(8, 8, mode="absolute", rngs=nnx.Rngs(0))
    layer = MonoResidual(8, 8, F=f, rngs=nnx.Rngs(0))
    assert layer.F is f


def test_F_and_explicit_subdepth_raises() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(
            8,
            8,
            F=MonoLinear(8, 8, mode="absolute", rngs=nnx.Rngs(0)),
            sub_depth=2,
            rngs=nnx.Rngs(0),
        )


def test_subdepth_below_one_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, 8, mode="absolute", sub_depth=0, rngs=nnx.Rngs(0))


def _nondecreasing(layer: MonoResidual, in_f: int) -> None:
    layer.alpha.value = jnp.array(0.3)
    layer.beta.value = jnp.array(0.7)
    import jax

    x = jax.random.normal(jax.random.key(1), (64, in_f))
    y0 = layer(x)
    for i in range(in_f):
        xp = x.at[:, i].add(0.5)
        assert float(jnp.min(layer(xp) - y0)) >= -1e-4


def test_monotone_identity_skip() -> None:
    _nondecreasing(
        MonoResidual(
            6, 6, mode="absolute", activation="elu", sub_depth=2, rngs=nnx.Rngs(0)
        ),
        6,
    )


def test_monotone_projection_skip() -> None:
    _nondecreasing(
        MonoResidual(
            6, 4, mode="switch", activation="elu", sub_depth=2, rngs=nnx.Rngs(0)
        ),
        6,
    )
