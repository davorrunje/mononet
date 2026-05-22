"""Contract test for the mononet.jax public API surface."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")


def test_mono_linear_exists_and_is_nnx_module() -> None:
    from mononet.jax import MonoLinear

    assert issubclass(MonoLinear, nnx.Module)


def test_mono_mlp_exists_and_is_nnx_module() -> None:
    from mononet.jax import MonoMLP

    assert issubclass(MonoMLP, nnx.Module)


def test_instantiating_mono_linear_raises_not_implemented() -> None:
    import numpy as np

    from mononet.core.types import ActivationSpec, MonotonicityMask
    from mononet.jax import MonoLinear

    with pytest.raises(NotImplementedError):
        MonoLinear(
            in_features=4,
            out_features=2,
            monotonicity=MonotonicityMask(np.zeros(4, dtype=np.int8)),
            activation=ActivationSpec(name="relu"),
            rngs=nnx.Rngs(0),
        )


def test_no_unexpected_top_level_exports() -> None:
    import mononet.jax as j

    expected = {"MonoLinear", "MonoMLP"}
    assert set(j.__all__) == expected
