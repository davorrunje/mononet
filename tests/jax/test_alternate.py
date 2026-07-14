# SPDX-License-Identifier: Apache-2.0
"""JAX (Flax NNX) tests for mode="alternate" + init-time prev=."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")

from typing import TYPE_CHECKING  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from mononet.jax import MonoLinear  # noqa: E402

if TYPE_CHECKING:
    from flax import nnx as nnx_types

    from mononet.core.types import ActivationName


def _stack(
    rngs: nnx_types.Rngs,
    act: ActivationName = "relu",
    depth: int = 4,
    d: int = 4,
    h: int = 16,
) -> nnx_types.Sequential:
    layers: list[MonoLinear] = []
    prev: MonoLinear | None = None
    prev_in = d
    for _ in range(depth):
        lay = MonoLinear(
            prev_in, h, mode="alternate", activation=act, prev=prev, rngs=rngs
        )
        layers.append(lay)
        prev, prev_in = lay, h
    layers.append(
        MonoLinear(prev_in, 1, mode="mixed", activation="identity", rngs=rngs)
    )
    return nnx.Sequential(*layers)  # type: ignore[no-any-return]


def _alternate_layers(net: nnx_types.Sequential) -> list[MonoLinear]:
    return [
        m for m in net.layers if isinstance(m, MonoLinear) and m.mode == "alternate"
    ]


def test_prev_alternates_phase_and_entry_is_convex() -> None:
    net = _stack(nnx.Rngs(0))
    alt = _alternate_layers(net)
    assert [m._alt_convex for m in alt] == [True, False, True, False]


def test_entry_bias_zero_interior_bias_alternates_sign() -> None:
    net = _stack(nnx.Rngs(0))
    alt = _alternate_layers(net)
    assert alt[0].bias is not None
    assert alt[1].bias is not None
    assert alt[2].bias is not None
    assert float(jnp.abs(alt[0].bias[...]).max()) == pytest.approx(0.0, abs=1e-6)
    assert float(alt[1].bias[...].mean()) < 0.0  # concave interior
    assert float(alt[2].bias[...].mean()) > 0.0  # convex interior


def test_prev_not_retained() -> None:
    net = _stack(nnx.Rngs(0))
    alt = _alternate_layers(net)
    assert all("prev" not in vars(m) for m in alt)
    state = nnx.state(net)
    flat = state.flat_state()
    assert not any("prev" in "/".join(str(p) for p in path) for path, _ in flat)
    assert not any("_alt" in "/".join(str(p) for p in path) for path, _ in flat)


def test_alternate_is_monotone_nondecreasing() -> None:
    net = _stack(nnx.Rngs(0))
    x = jnp.zeros((1, 4))
    base = net(x)
    for j in range(4):
        bumped = x.at[0, j].add(1e-2)
        assert float((net(bumped) - base).reshape(())) >= -1e-5


def test_convex_fraction_rejected_for_alternate() -> None:
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoLinear(
            4,
            8,
            mode="alternate",
            activation="relu",
            convex_fraction=0.3,
            rngs=nnx.Rngs(0),
        )


def test_prev_rejected_for_non_alternate() -> None:
    rngs = nnx.Rngs(0)
    entry = MonoLinear(4, 8, mode="alternate", activation="relu", rngs=rngs)
    with pytest.raises(ValueError, match="prev"):
        MonoLinear(8, 8, mode="mixed", activation="relu", prev=entry, rngs=rngs)


def test_prev_must_be_alternate() -> None:
    rngs = nnx.Rngs(0)
    mixed = MonoLinear(4, 8, mode="mixed", activation="relu", rngs=rngs)
    with pytest.raises(ValueError, match="alternate"):
        MonoLinear(8, 8, mode="alternate", activation="relu", prev=mixed, rngs=rngs)


def test_deep_alternate_trains_stably() -> None:
    # depth-8 plain alternate stack does not diverge (contrast: mixed diverges).
    rng = np.random.default_rng(0)
    x_np = rng.uniform(-1, 1, (2000, 4))
    x = jnp.asarray(x_np, dtype=jnp.float32)
    y_np = (1 / (1 + np.exp(-3 * (x_np - 0.1)))).sum(1, keepdims=True)
    y_np = (y_np - y_np.mean()) / y_np.std()
    y = jnp.asarray(y_np, dtype=jnp.float32)
    net = _stack(nnx.Rngs(0), depth=8)
    optimizer = nnx.Optimizer(net, optax.adam(1e-2), wrt=nnx.Param)

    def loss_fn(
        model: nnx_types.Sequential, x: jnp.ndarray, y: jnp.ndarray
    ) -> jnp.ndarray:
        pred = model(x)
        return jnp.mean((pred - y) ** 2)

    for _ in range(300):
        _loss, grads = nnx.value_and_grad(loss_fn)(net, x, y)
        optimizer.update(net, grads)
    assert float(loss_fn(net, x, y)) < 0.9  # beats predict-the-mean (~1.0)


def test_mono_residual_rejects_alternate() -> None:
    from mononet.jax import MonoResidual

    with pytest.raises(ValueError, match="alternate"):
        MonoResidual(8, 8, mode="alternate", activation="relu", rngs=nnx.Rngs(0))
