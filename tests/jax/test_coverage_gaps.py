# SPDX-License-Identifier: Apache-2.0
"""Cover jax kernel error paths and layer branches not hit elsewhere."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402

from mononet.jax import (  # noqa: E402
    MonoInput,
    MonoLinear,
    MonoResidual,
    _kernels,
)


def test_kernel_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _kernels.activation("bogus", jnp.zeros(3))


def test_kernel_gate_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        _kernels.gate("bogus", jnp.zeros(()))


def test_kernel_dense_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _kernels.monotonic_dense(
            jnp.zeros((2, 3)), jnp.ones((3, 4)), jnp.zeros(4), "bogus", "relu"
        )


def test_residual_accepts_callable_f_factory() -> None:
    rngs = nnx.Rngs(0)
    # A plain callable (not an nnx.Module) hits the `self.F = F(units)` branch.
    block = MonoResidual(
        4, 4, F=lambda u: MonoLinear(u, u, activation="relu", rngs=rngs), rngs=rngs
    )
    assert block(jnp.zeros((2, 4))).shape == (2, 4)


def test_mono_input_accepts_scalar_direction() -> None:
    layer = MonoInput(-1)
    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(layer(x), -x)
