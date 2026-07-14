"""Near-zero init of the default F, and gate defaults, for MonoResidual (jax)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402

from mononet.jax import MonoLinear, MonoResidual  # noqa: E402


def _last_linear(block: MonoResidual) -> MonoLinear:
    f = block.F
    if isinstance(f, MonoLinear):
        return f
    # f is nnx.Sequential here; mypy doesn't narrow unions of nnx.Module
    assert hasattr(f, "layers"), f"Expected f to have 'layers', got {type(f)}"
    last = f.layers[-1]  # type: ignore[attr-defined]
    assert isinstance(last, MonoLinear)
    return last


def test_default_F_last_layer_is_near_zero_but_nonzero() -> None:  # noqa: N802
    block = MonoResidual(32, 32, mode="mixed", activation="elu", rngs=nnx.Rngs(0))
    last = _last_linear(block)
    wnorm = float(jnp.abs(last.weight[...]).sum())
    # small but NOT exactly zero (exact zero would freeze under |W|)
    assert wnorm > 0.0
    assert wnorm < 1.0  # heavily attenuated vs a normal init (~tens)
    # bias zeroed
    assert last.bias is not None
    assert float(jnp.abs(last.bias[...]).sum()) == 0.0


def test_default_block_is_near_identity_at_init() -> None:
    block = MonoResidual(32, 32, mode="mixed", activation="elu", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (8, 32))
    fx_rms = float(jnp.sqrt(jnp.mean(block.F(x) ** 2)))
    assert fx_rms < 0.2  # F(x) ~= 0 at init => block ~= g_alpha * skip


def test_custom_F_is_not_near_zeroed() -> None:  # noqa: N802
    custom = MonoLinear(32, 32, mode="mixed", activation="elu", rngs=nnx.Rngs(0))
    before = float(jnp.abs(custom.weight[...]).sum())
    block = MonoResidual(32, 32, F=custom, rngs=nnx.Rngs(1))
    after = float(jnp.abs(block.F.weight[...]).sum())  # type: ignore[attr-defined]
    assert after == before  # untouched


def test_near_zero_scale_is_user_tunable() -> None:
    small = _last_linear(
        MonoResidual(32, 32, mode="mixed", activation="elu", rngs=nnx.Rngs(0))
    )
    big = _last_linear(
        MonoResidual(
            32,
            32,
            mode="mixed",
            activation="elu",
            near_zero_scale=2e-3,
            rngs=nnx.Rngs(0),
        )
    )
    # same seed => 2e-3 gives ~2x the weight magnitude of the 1e-3 default
    ratio = float(jnp.abs(big.weight[...]).sum()) / float(
        jnp.abs(small.weight[...]).sum()
    )
    assert ratio == pytest.approx(2.0, rel=1e-5)
    # 0.0 reproduces exact-zero
    zero = _last_linear(
        MonoResidual(
            32,
            32,
            mode="mixed",
            activation="elu",
            near_zero_scale=0.0,
            rngs=nnx.Rngs(0),
        )
    )
    assert float(jnp.abs(zero.weight[...]).sum()) == 0.0


def test_near_zero_scale_with_bias_false() -> None:
    # covers the no-bias branch of near-zero init: weight scaled, no bias to zero
    layer = MonoLinear(
        4,
        4,
        mode="mixed",
        activation="elu",
        bias=False,
        near_zero_scale=1e-3,
        rngs=nnx.Rngs(0),
    )
    assert layer.bias is None
    assert float(jnp.abs(layer.weight[...]).sum()) > 0.0  # scaled but nonzero
    layer(jnp.zeros((2, 4)))  # forward runs without a bias
