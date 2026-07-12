# SPDX-License-Identifier: Apache-2.0
"""JAX (Flax NNX) model builders for the four PINN methods.

Mirrors the PyTorch builders: a callable ``u(x, t)`` on column vectors, with the
monotone-in-``x`` / free-in-``t`` structure from embedding ``t`` through an
unconstrained MLP and feeding ``[x, h(t)]`` to a stack monotone in all inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mononet.core.types import ActivationSpec, MonotonicityMask
from mononet.jax import MonoInput, MonoLinear, MonoResidual

if TYPE_CHECKING:
    from applications.pinn.models.protocol import Method, ModelConfig

_ACTS = {"tanh": jnp.tanh, "elu": jax.nn.elu, "softplus": jax.nn.softplus}


class _PlainMLP(nnx.Module):
    """Unconstrained two-hidden-layer MLP."""

    def __init__(
        self, in_dim: int, width: int, out_dim: int, activation: str, *, rngs: nnx.Rngs
    ) -> None:
        self.l1 = nnx.Linear(in_dim, width, rngs=rngs)
        self.l2 = nnx.Linear(width, width, rngs=rngs)
        self.l3 = nnx.Linear(width, out_dim, rngs=rngs)
        self.act = _ACTS[activation]

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the MLP."""
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        return self.l3(x)


class _ClampedLinear(nnx.Module):
    """Linear layer with non-negative weights (inexpressive monotone map)."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs) -> None:
        key = rngs.params()
        scale = (2.0 / (in_features + out_features)) ** 0.5
        self.w = nnx.Param(jax.random.normal(key, (in_features, out_features)) * scale)
        self.b = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply ``x @ |W| + b``."""
        return x @ jnp.abs(self.w.get_value()) + self.b.get_value()


class VanillaMLP(nnx.Module):
    """Unconstrained MLP over ``[x, t]`` (also the ``soft`` architecture)."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs) -> None:
        """Build the unconstrained MLP."""
        self.net = _PlainMLP(2, cfg.width, 1, cfg.plain_activation, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Return ``u(x, t)``."""
        return self.net(jnp.concatenate([x, t], axis=-1))


class WeightClipMono(nnx.Module):
    """Monotone-in-``x`` net via non-negative weights (inexpressive baseline)."""

    def __init__(self, cfg: ModelConfig, sign_x: int, *, rngs: nnx.Rngs) -> None:
        """Build the clamped-weight monotone stack and the ``t`` embedding."""
        self.sign_x = float(sign_x)
        self.t_embed = _PlainMLP(
            1, cfg.t_embed_width, cfg.t_embed_dim, cfg.plain_activation, rngs=rngs
        )
        in_dim = 1 + cfg.t_embed_dim
        self.c1 = _ClampedLinear(in_dim, cfg.width, rngs=rngs)
        self.c2 = _ClampedLinear(cfg.width, cfg.width, rngs=rngs)
        self.c3 = _ClampedLinear(cfg.width, 1, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Return ``u(x, t)`` (monotone in ``x`` via non-negative weights)."""
        z = jnp.concatenate([self.sign_x * x, self.t_embed(t)], axis=-1)
        z = jax.nn.softplus(self.c1(z))
        z = jax.nn.softplus(self.c2(z))
        return self.c3(z)


class HardMonoField(nnx.Module):
    """Expressive hard-monotone-in-``x`` field built from ``mononet`` layers."""

    def __init__(self, cfg: ModelConfig, sign_x: int, *, rngs: nnx.Rngs) -> None:
        """Build the mononet monotone stack and the unconstrained ``t`` embedding."""
        self.t_embed = _PlainMLP(
            1, cfg.t_embed_width, cfg.t_embed_dim, cfg.plain_activation, rngs=rngs
        )
        in_dim = 1 + cfg.t_embed_dim
        mask = MonotonicityMask(
            np.array([sign_x, *([1] * cfg.t_embed_dim)], dtype=np.int8)
        )
        self.mono_input = MonoInput(mask)
        act = ActivationSpec(cfg.mono_activation)  # type: ignore[arg-type]
        self.n_blocks = cfg.n_blocks
        # nnx tracks Module-valued attributes; a plain list is not a pytree node,
        # so register each block under its own attribute name.
        self.block0 = MonoResidual(
            in_dim,
            cfg.width,
            mode=cfg.mode,
            activation=act,
            rngs=rngs,  # type: ignore[arg-type]
        )
        for i in range(1, cfg.n_blocks):
            setattr(
                self,
                f"block{i}",
                MonoResidual(
                    cfg.width, cfg.width, mode=cfg.mode, activation=act, rngs=rngs
                ),  # type: ignore[arg-type]
            )
        self.head = MonoLinear(
            cfg.width, 1, mode=cfg.mode, activation="identity", rngs=rngs
        )  # type: ignore[arg-type]

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Return ``u(x, t)`` (monotone in ``x`` by construction)."""
        z = jnp.concatenate([x, self.t_embed(t)], axis=-1)
        z = self.mono_input(z)
        for i in range(self.n_blocks):
            z = getattr(self, f"block{i}")(z)
        return self.head(z)


def build_jax(problem: object, cfg: ModelConfig, method: Method) -> nnx.Module:
    """Build a JAX (Flax NNX) model for ``method``.

    :param problem: A registered `Problem` (its admissibility mask sets ``sign_x``).
    :param cfg: Architecture configuration.
    :param method: One of ``vanilla`` / ``soft`` / ``weight_clip`` / ``hard_monotone``.
    :returns: A Flax NNX module callable as ``u(x, t)``.
    :raises ValueError: If ``method`` is unknown.
    """
    rngs = nnx.Rngs(cfg.seed)
    sign_x = int(problem.admissibility().mask[0])  # type: ignore[attr-defined]
    if method in ("vanilla", "soft"):
        return VanillaMLP(cfg, rngs=rngs)
    if method == "weight_clip":
        return WeightClipMono(cfg, sign_x, rngs=rngs)
    if method == "hard_monotone":
        return HardMonoField(cfg, sign_x, rngs=rngs)
    raise ValueError(f"unknown method {method!r}")
