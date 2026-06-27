"""JAX (Flax NNX) idiomatic layer wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.nn.initializers as jinit
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask
from mononet.jax import _kernels

if TYPE_CHECKING:
    from collections.abc import Callable

_INIT_FNS = {
    "he_normal": jinit.he_normal(),
    "glorot_uniform": jinit.glorot_uniform(),
    "lecun_normal": jinit.lecun_normal(),
}


def _act_name(activation: ActivationSpec | str) -> str:
    """Extract activation name string from spec or plain string.

    :param activation: Either a string name or an :class:`ActivationSpec`.
    :returns: Activation name string.
    """
    return activation if isinstance(activation, str) else activation.name


def _init_array(
    shape: tuple[int, int], init: InitSpec | str | None, rngs: nnx.Rngs
) -> jnp.ndarray:
    """Initialise a weight array using the given spec and RNG state.

    :param shape: Desired ``(in_features, units)`` shape.
    :param init: Initializer spec, name string, or ``None`` for ``he_normal``.
    :param rngs: Flax NNX RNG container.
    :returns: Initialised JAX array of ``shape``.
    """
    spec = (
        InitSpec()
        if init is None
        else (InitSpec(scheme=init) if isinstance(init, str) else init)  # type: ignore[arg-type]
    )
    return _INIT_FNS[spec.scheme](rngs.params(), shape)


class MonoLinear(nnx.Module):
    """Monotonic analogue of ``flax.nnx.Linear`` (non-decreasing in all inputs).

    :param in_features: Number of input features.
    :param units: Number of output units.
    :param mode: ``switch`` (default) or ``absolute``.
    :param activation: Base activation; one of ``relu``, ``elu``, ``selu``,
        ``softplus``.
    :param convex_fraction: Fraction of convex units (``absolute`` mode only).
    :param init: Weight initializer; defaults to ``he_normal``.
    :param bias: Whether to include a bias vector.
    :param rngs: Flax NNX RNG container.
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        *,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        convex_fraction: float = 0.5,
        init: InitSpec | str | None = None,
        bias: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise MonoLinear with weights and optional bias."""
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.convex_fraction = convex_fraction
        self.weight = nnx.Param(_init_array((in_features, units), init, rngs))
        self.bias: nnx.Param[jnp.ndarray] | None = (
            nnx.Param(jnp.zeros((units,))) if bias else None
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the monotonic dense transformation.

        :param x: Input tensor of shape ``(batch, in_features)``.
        :returns: Output tensor of shape ``(batch, units)``.
        """
        bias = (
            self.bias[...]
            if self.bias is not None
            else jnp.zeros((self.weight[...].shape[1],), dtype=x.dtype)
        )
        return _kernels.monotonic_dense(
            x,
            self.weight[...],
            bias,
            self.mode,
            self.activation_name,
            self.convex_fraction,
        )


class MonoResidual(nnx.Module):
    """Dual-gated monotone residual block (Flax NNX).

    :param in_features: Number of input features.
    :param units: Number of output units.
    :param F: Inner monotone sublayer; defaults to a single
        :class:`MonoLinear`. May also be a callable ``(units) -> Module``.
    :param mode: ``switch`` or ``absolute``.
    :param activation: Base activation name or spec.
    :param alpha_gate: Gate token for the skip path; default ``shifted_elu``.
    :param beta_gate: Gate token for the dense path; default ``scaled_elu``.
    :param init: Weight initializer; defaults to ``he_normal``.
    :param rngs: Flax NNX RNG container.
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        *,
        F: nnx.Module | Callable[[int], nnx.Module] | None = None,  # noqa: N803
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "scaled_elu",
        init: InitSpec | str | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise MonoResidual with sublayer F and scalar gate params."""
        if F is None:
            self.F: nnx.Module = MonoLinear(
                in_features,
                units,
                mode=mode,
                activation=activation,
                init=init,
                rngs=rngs,
            )
        elif callable(F) and not isinstance(F, nnx.Module):
            self.F = F(units)
        else:
            self.F = F
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        self.alpha = nnx.Param(jnp.zeros(()))
        self.beta = nnx.Param(jnp.zeros(()))
        if in_features == units:
            self.skip_weight: nnx.Param[jnp.ndarray] | None = None
        else:
            self.skip_weight = nnx.Param(_init_array((in_features, units), init, rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply ``g_alpha * skip(x) + g_beta * F(x)``.

        :param x: Input tensor of shape ``(batch, in_features)``.
        :returns: Output tensor of shape ``(batch, units)``.
        """
        skip = x if self.skip_weight is None else x @ jnp.exp(self.skip_weight[...])
        fx: jnp.ndarray = self.F(x)  # type: ignore[assignment]
        return (
            _kernels.gate(self.alpha_gate, self.alpha[...]) * skip
            + _kernels.gate(self.beta_gate, self.beta[...]) * fx
        )


class MonoInput(nnx.Module):
    """Sign-flip layer mapping prescribed directions onto non-decreasing layers.

    :param directions: Either a scalar ``+1``/``-1``, or a
        :class:`~mononet.core.types.MonotonicityMask` giving per-feature
        directions.
    """

    def __init__(self, directions: int | MonotonicityMask) -> None:
        """Initialise MonoInput with a fixed directions array."""
        if isinstance(directions, MonotonicityMask):
            d = directions.values.astype(np.float32)
        else:
            d = np.array(float(directions), dtype=np.float32)
        self.directions = nnx.Variable(jnp.asarray(d))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Negate ``-1`` columns; pass ``+1`` columns through.

        :param x: Input tensor of shape ``(batch, features)``.
        :returns: Sign-flipped tensor of the same shape.
        """
        return x * self.directions[...].astype(x.dtype)
