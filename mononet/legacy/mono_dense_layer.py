# SPDX-License-Identifier: Apache-2.0
"""Legacy MonoDense layer for migration from airtai/monotonic-nn.

A faithful, backend-agnostic reproduction of the original ``airtai/monotonic-nn``
implementation using ``keras.ops`` so it runs under any Keras 3 backend. Reproduces
the original public API (``MonoDense`` plus builders and helpers) as a migration
bridge; new code should use ``mononet.torch`` / ``mononet.jax`` / ``mononet.keras``
instead. Every ``MonoDense`` construction emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import keras
import numpy as np
from keras import activations, ops

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_WARNED = False

_DEPRECATION_MESSAGE = (
    "mononet.legacy.MonoDense reproduces the original airtai/monotonic-nn layer "
    "for migration only. Prefer mononet.torch/jax/keras (MonoLinear/MonoDense). "
    "Note the monotonicity spec changed: the legacy {-1, 0, 1} indicator maps to "
    "a two-value +/-1 mask in the new layers."
)


def _warn_once() -> None:
    """Emit the legacy :class:`DeprecationWarning` at most once per process."""
    global _WARNED
    if not _WARNED:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        _WARNED = True


def get_saturated_activation(
    convex_activation: Callable[[Any], Any],
    concave_activation: Callable[[Any], Any],
    a: float = 1.0,
    c: float = 1.0,
) -> Callable[[Any], Any]:
    """Build the saturated activation from a convex/concave pair.

    :param convex_activation: Convex, monotonically increasing base activation.
    :param concave_activation: Its concave reflection ``-f(-x)``.
    :param a: Output scale.
    :param c: Knot location of the piecewise join.
    :returns: A callable mapping a tensor to the saturated activation.
    """

    def saturated_activation(x: Any) -> Any:
        cc = convex_activation(ops.ones_like(x) * c)
        return a * ops.where(
            x <= 0,
            convex_activation(x + c) - cc,
            concave_activation(x - c) + cc,
        )

    return saturated_activation


@lru_cache
def get_activation_functions(
    activation: str | Callable[[Any], Any] | None = None,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any]]:
    """Resolve the convex, concave, and saturated activations for a base name.

    :param activation: Base activation name or callable (assumed convex,
        monotonically increasing). ``None`` resolves to the linear activation.
    :returns: ``(convex_activation, concave_activation, saturated_activation)``.
    """
    convex_activation = activations.get(
        activation.lower() if isinstance(activation, str) else activation
    )

    def concave_activation(x: Any) -> Any:
        return -convex_activation(-x)

    saturated_activation = get_saturated_activation(
        convex_activation, concave_activation
    )
    return convex_activation, concave_activation, saturated_activation


def apply_activations(
    x: Any,
    *,
    units: int,
    convex_activation: Callable[[Any], Any],
    concave_activation: Callable[[Any], Any],
    saturated_activation: Callable[[Any], Any],
    is_convex: bool = False,
    is_concave: bool = False,
    activation_weights: tuple[float, float, float] = (7.0, 7.0, 2.0),
) -> Any:
    """Split ``x`` into convex/concave/saturated groups and activate each.

    :param x: Pre-activation tensor of shape ``(batch, units)``.
    :param units: Output width (the size of the last axis of ``x``).
    :param convex_activation: Convex branch activation.
    :param concave_activation: Concave branch activation.
    :param saturated_activation: Saturated branch activation.
    :param is_convex: Force an all-convex split ``(units, 0, 0)``.
    :param is_concave: Force an all-concave split ``(0, units, 0)``.
    :param activation_weights: Relative sizes of the three groups; ignored when
        ``is_convex`` or ``is_concave`` is set.
    :returns: Activated tensor of shape ``(batch, units)``.
    :raises ValueError: If ``activation_weights`` is not length 3 or has a
        negative entry.
    """
    if convex_activation is None:
        return x

    if is_convex:
        normalized_activation_weights = np.array([1.0, 0.0, 0.0])
    elif is_concave:
        normalized_activation_weights = np.array([0.0, 1.0, 0.0])
    else:
        if len(activation_weights) != 3:
            raise ValueError(f"activation_weights={activation_weights}")
        if (np.array(activation_weights) < 0).any():
            raise ValueError(f"activation_weights={activation_weights}")
        normalized_activation_weights = np.array(activation_weights) / sum(
            activation_weights
        )

    s_convex = round(normalized_activation_weights[0] * units)
    s_concave = round(normalized_activation_weights[1] * units)

    # keras.ops.split takes cut points (numpy semantics), not sizes.
    x_convex, x_concave, x_saturated = ops.split(
        x, [s_convex, s_convex + s_concave], axis=-1
    )

    y_convex = convex_activation(x_convex)
    y_concave = concave_activation(x_concave)
    y_saturated = saturated_activation(x_saturated)

    return ops.concatenate([y_convex, y_concave, y_saturated], axis=-1)


def get_monotonicity_indicator(
    monotonicity_indicator: Any,
    *,
    input_shape: tuple[int | None, ...],
    units: int,
) -> np.ndarray:
    """Normalise a monotonicity indicator to a broadcastable column vector.

    :param monotonicity_indicator: Scalar or array of per-input signs, each in
        ``{-1, 0, 1}`` (``1`` increasing, ``-1`` decreasing, ``0`` free).
    :param input_shape: Layer input shape; ``input_shape[-1]`` is the fan-in.
    :param units: Output width.
    :returns: The indicator reshaped to ``(fan_in, 1)`` (or as given if already
        2-D), validated against ``{-1, 0, 1}``.
    :raises ValueError: If the indicator has rank > 2 or contains a value
        outside ``{-1, 0, 1}``.
    """
    ind = np.array(monotonicity_indicator)
    if len(ind.shape) < 2:
        ind = np.reshape(ind, (-1, 1))
    elif len(ind.shape) > 2:
        raise ValueError(f"monotonicity_indicator has rank greater than 2: {ind.shape}")

    fan_in = cast("int", input_shape[-1])
    np.broadcast_to(ind, shape=(fan_in, units))

    if not np.all((ind == -1) | (ind == 0) | (ind == 1)):
        raise ValueError(
            "Each element of monotonicity_indicator must be one of -1, 0, 1, "
            f"but it is: '{ind}'"
        )
    return ind


def apply_monotonicity_indicator_to_kernel(
    kernel: Any,
    monotonicity_indicator: Any,
) -> Any:
    """Sign-constrain a kernel by a monotonicity indicator.

    :param kernel: Weight tensor of shape ``(fan_in, units)``.
    :param monotonicity_indicator: Broadcastable ``{-1, 0, 1}`` indicator.
    :returns: Kernel with ``|W|`` where the indicator is ``1``, ``-|W|`` where
        it is ``-1``, and ``W`` unchanged where it is ``0``.
    """
    monotonicity_indicator = ops.convert_to_tensor(monotonicity_indicator)
    abs_kernel = ops.abs(kernel)
    xs = ops.where(monotonicity_indicator == 1, abs_kernel, kernel)
    xs = ops.where(monotonicity_indicator == -1, -abs_kernel, xs)
    return xs


@contextmanager
def replace_kernel_using_monotonicity_indicator(
    layer: Any,
    monotonicity_indicator: Any,
) -> Generator[None]:
    """Temporarily swap ``layer.kernel`` for its sign-constrained version.

    Retained for API compatibility with the original package. The ported
    :class:`MonoDense` does not rely on this context manager (it constrains the
    kernel functionally in ``call`` instead).

    :param layer: A layer exposing a mutable ``kernel`` attribute.
    :param monotonicity_indicator: Broadcastable ``{-1, 0, 1}`` indicator.
    :yields: Nothing; restores the original kernel on exit.
    """
    old_kernel = layer.kernel
    layer.kernel = apply_monotonicity_indicator_to_kernel(
        layer.kernel, monotonicity_indicator
    )
    try:
        yield
    finally:
        layer.kernel = old_kernel


class MonoDense(keras.layers.Dense):  # type: ignore[misc]
    """Monotonic counterpart of ``keras.layers.Dense`` (legacy API).

    Faithful reproduction of the original ``airtai/monotonic-nn`` layer. The
    kernel is sign-constrained per :paramref:`monotonicity_indicator` and the
    output is split into convex/concave/saturated activation groups.

    :param units: Output dimensionality.
    :param activation: Base activation (assumed convex, monotonically
        increasing), e.g. ``"relu"`` or ``"elu"``; name or callable. ``None``
        is linear.
    :param monotonicity_indicator: Per-input sign in ``{-1, 0, 1}`` (``1``
        increasing, ``-1`` decreasing, ``0`` non-monotonic). Scalar or array.
    :param is_convex: Force an all-convex activation split.
    :param is_concave: Force an all-concave activation split.
    :param activation_weights: Relative sizes of the convex/concave/saturated
        groups; ignored when ``is_convex`` or ``is_concave`` is set.
    :raises ValueError: If both ``is_convex`` and ``is_concave`` are set, or if
        ``activation_weights`` is not length 3 or has a negative entry.
    """

    def __init__(
        self,
        units: int,
        *,
        activation: str | Callable[[Any], Any] | None = None,
        monotonicity_indicator: Any = 1,
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: tuple[float, float, float] = (7.0, 7.0, 2.0),
        **kwargs: Any,
    ) -> None:
        """Construct a legacy MonoDense layer (emits a DeprecationWarning)."""
        _warn_once()
        if is_convex and is_concave:
            raise ValueError(
                "The model cannot be set to be both convex and concave "
                "(only linear functions are both)."
            )
        if len(activation_weights) != 3:
            raise ValueError(
                "There must be exactly three components of activation_weights, "
                f"but we have this instead: {activation_weights}."
            )
        if (np.array(activation_weights) < 0).any():
            raise ValueError(
                "Values of activation_weights must be non-negative, but we have "
                f"this instead: {activation_weights}."
            )

        super().__init__(units=units, activation=None, **kwargs)

        self.units = units
        self.org_activation = activation
        self.monotonicity_indicator = monotonicity_indicator
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = activation_weights

        (
            self.convex_activation,
            self.concave_activation,
            self.saturated_activation,
        ) = get_activation_functions(self.org_activation)

    def build(self, input_shape: Any) -> None:
        """Create the Dense weights and normalise the indicator.

        :param input_shape: Shape tuple; ``input_shape[-1]`` is the fan-in.
        """
        super().build(input_shape)
        self.monotonicity_indicator = get_monotonicity_indicator(
            monotonicity_indicator=self.monotonicity_indicator,
            input_shape=input_shape,
            units=self.units,
        )

    def call(self, inputs: Any) -> Any:
        """Apply the sign-constrained affine map and grouped activations.

        :param inputs: Input tensor of shape ``(batch, ..., fan_in)``.
        :returns: Output tensor of shape ``(batch, ..., units)``.
        """
        constrained_kernel = apply_monotonicity_indicator_to_kernel(
            ops.convert_to_tensor(self.kernel), self.monotonicity_indicator
        )
        h = ops.matmul(inputs, constrained_kernel)
        if self.use_bias:
            h = h + self.bias
        return apply_activations(
            h,
            units=self.units,
            convex_activation=self.convex_activation,
            concave_activation=self.concave_activation,
            saturated_activation=self.saturated_activation,
            is_convex=self.is_convex,
            is_concave=self.is_concave,
            activation_weights=self.activation_weights,
        )

    def get_config(self) -> dict[str, Any]:
        """Serialize the layer configuration.

        :returns: Config dict with the original legacy keys.
        """
        return {
            "units": self.units,
            "activation": self.org_activation,
            "monotonicity_indicator": self.monotonicity_indicator,
            "is_convex": self.is_convex,
            "is_concave": self.is_concave,
            "activation_weights": self.activation_weights,
        }
