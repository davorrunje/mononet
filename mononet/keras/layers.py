# SPDX-License-Identifier: Apache-2.0
"""Keras 3 idiomatic layer wrappers."""

from __future__ import annotations

from typing import Any, cast

import keras
import numpy as np
from keras import ops

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask
from mononet.keras import _kernels


def _act_name(activation: ActivationSpec | str) -> str:
    """Return the string name of an activation spec or pass through a string.

    :param activation: Activation spec or name string.
    :returns: Activation name string.
    """
    return activation if isinstance(activation, str) else activation.name


def _init_name(init: InitSpec | str | None) -> str:
    """Return the string initializer name, defaulting to ``he_normal``.

    :param init: Init spec, name string, or ``None``.
    :returns: Keras initializer name string.
    """
    if init is None:
        return "he_normal"
    return init if isinstance(init, str) else init.scheme


class MonoDense(keras.layers.Layer):  # type: ignore[misc]
    """Monotonic analogue of ``keras.layers.Dense`` (non-decreasing in all inputs).

    The weight matrix is constrained at call-time (not at parameter-update time)
    by the ``switch`` or ``absolute`` mode, as described in the paper.

    :param units: Output dimensionality.
    :param mode: One of ``switch`` (default) or ``absolute``.
    :param activation: Base activation name or :class:`~mononet.core.types.ActivationSpec`.
    :param convex_fraction: Fraction of output units using the convex branch
        (only used in ``absolute`` mode).
    :param init: Initializer name or :class:`~mononet.core.types.InitSpec`.
    :param bias: Whether to include a bias term (default ``True``).
    """

    def __init__(
        self,
        units: int,
        *,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        convex_fraction: float = 0.5,
        init: InitSpec | str | None = None,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise MonoDense."""
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.convex_fraction = convex_fraction
        self.init_name = _init_name(init)
        self.use_bias = bias

    def build(self, input_shape: Any) -> None:
        """Create weights once the input width is known.

        :param input_shape: Shape tuple; ``input_shape[-1]`` is ``in_features``.
        """
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer=self.init_name,
            trainable=True,
            name="weight",
        )
        self.b = (
            self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
            if self.use_bias
            else None
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply the monotonic dense transformation.

        :param inputs: Input tensor of shape ``(batch, in_features)``.
        :returns: Output tensor of shape ``(batch, units)``.
        """
        bias = self.b if self.b is not None else ops.zeros((self.units,))
        return _kernels.monotonic_dense(
            inputs,
            self.w,
            bias,
            self.mode,
            self.activation_name,
            self.convex_fraction,
        )

    def get_config(self) -> dict[str, Any]:
        """Serialize token/scalar fields (callables are not serializable).

        :returns: Config dict suitable for :meth:`from_config`.
        """
        cfg = cast("dict[str, Any]", super().get_config())
        cfg.update(
            {
                "units": self.units,
                "mode": self.mode,
                "activation": self.activation_name,
                "convex_fraction": self.convex_fraction,
                "init": self.init_name,
                "bias": self.use_bias,
            }
        )
        return cfg


class MonoResidual(keras.layers.Layer):  # type: ignore[misc]
    """Dual-gated monotone residual block (Keras 3).

    Computes ``g_a * skip(x) + g_b * F(x)`` where gates are initialized to
    yield ``~identity`` (warm start).

    :param units: Output width; must equal input width if no skip projection is
        desired.
    :param F: Inner monotone layer; defaults to a fresh :class:`MonoDense`.
    :param mode: Forwarded to the default ``F``.
    :param activation: Forwarded to the default ``F``.
    :param alpha_gate: Gate token for the skip path (``shifted_elu``).
    :param beta_gate: Gate token for the dense path (``scaled_elu``).
    :param init: Forwarded to the default ``F``.
    """

    def __init__(
        self,
        units: int,
        *,
        F: keras.layers.Layer | None = None,  # noqa: N803
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "scaled_elu",
        init: InitSpec | str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise MonoResidual."""
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.init_name = _init_name(init)
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        self.F = (
            F
            if F is not None
            else MonoDense(units, mode=mode, activation=activation, init=init)
        )

    def build(self, input_shape: Any) -> None:
        """Create gate scalars and the projection shortcut if needed.

        :param input_shape: Shape tuple; ``input_shape[-1]`` is ``in_features``.
        """
        in_features = int(input_shape[-1])
        self.alpha = self.add_weight(
            shape=(), initializer="zeros", trainable=True, name="alpha"
        )
        self.beta = self.add_weight(
            shape=(), initializer="zeros", trainable=True, name="beta"
        )
        self.skip_w: Any = (
            None
            if in_features == self.units
            else self.add_weight(
                shape=(in_features, self.units),
                initializer=self.init_name,
                trainable=True,
                name="skip_weight",
            )
        )
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply ``g_a * skip(x) + g_b * F(x)``.

        :param inputs: Input tensor of shape ``(batch, in_features)``.
        :returns: Output tensor of shape ``(batch, units)``.
        """
        skip = (
            inputs if self.skip_w is None else ops.matmul(inputs, ops.exp(self.skip_w))
        )
        return _kernels.gate(self.alpha_gate, self.alpha) * skip + _kernels.gate(
            self.beta_gate, self.beta
        ) * self.F(inputs)

    def get_config(self) -> dict[str, Any]:
        """Serialize token/scalar fields.

        :returns: Config dict suitable for :meth:`from_config`.
        """
        cfg = cast("dict[str, Any]", super().get_config())
        cfg.update(
            {
                "units": self.units,
                "mode": self.mode,
                "activation": self.activation_name,
                "alpha_gate": self.alpha_gate,
                "beta_gate": self.beta_gate,
                "init": self.init_name,
            }
        )
        return cfg


class MonoInput(keras.layers.Layer):  # type: ignore[misc]
    """Sign-flip layer mapping prescribed directions onto non-decreasing layers.

    :param directions: Either an integer scalar (``+1`` or ``-1``) applied to
        all inputs, a :class:`~mononet.core.types.MonotonicityMask` with
        per-feature signs, or a ``list[float]`` of per-feature signs (the
        serialized form produced by :meth:`get_config`).
    """

    def __init__(
        self, directions: int | MonotonicityMask | list[float], **kwargs: Any
    ) -> None:
        """Initialise MonoInput."""
        super().__init__(**kwargs)
        self._directions: float | list[float]
        if isinstance(directions, MonotonicityMask):
            self._directions = directions.values.astype(np.float32).tolist()
        elif isinstance(directions, list):
            self._directions = directions
        else:
            self._directions = float(directions)

    def call(self, inputs: Any) -> Any:
        """Negate ``-1`` columns; pass ``+1`` columns through.

        :param inputs: Input tensor of shape ``(batch, features)``.
        :returns: Sign-flipped tensor with the same shape.
        """
        return inputs * ops.convert_to_tensor(self._directions, dtype=inputs.dtype)

    def get_config(self) -> dict[str, Any]:
        """Serialize the direction spec.

        :returns: Config dict suitable for :meth:`from_config`.
        """
        cfg = cast("dict[str, Any]", super().get_config())
        cfg.update({"directions": self._directions})
        return cfg
