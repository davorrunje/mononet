# SPDX-License-Identifier: Apache-2.0
"""Keras 3 idiomatic layer wrappers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import keras
import numpy as np
from keras import ops

from mononet.core.init import (
    absolute_init_params,
    alternating_init_params,
    alternating_weight_bias,
)
from mononet.core.types import (
    ActivationName,
    ActivationSpec,
    InitSpec,
    MonotonicityMask,
)
from mononet.keras import _kernels

if TYPE_CHECKING:
    from mononet.core.config import Mode


_NEAR_ZERO_SCALE = 1e-3


def _act_name(activation: ActivationSpec | ActivationName) -> str:
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
    by the ``split`` or ``mixed`` mode, as described in the paper.

    :param units: Output dimensionality.
    :param mode: One of ``mixed`` (default), ``split``, or ``alternate``.
        Under ``alternate`` the layer is a *pure* (all-convex or all-concave)
        ``|W|`` layer whose phase and incoming mean are derived from ``prev``
        — see ``prev``.
    :param activation: Base activation, one of ``"relu"``, ``"elu"``,
        ``"selu"``, ``"softplus"``, ``"identity"``, or an
        :class:`~mononet.core.types.ActivationSpec`. ``None`` (the default)
        means ``"identity"`` — a linear monotone map, matching
        ``keras.layers.Dense``.
    :param convex_fraction: Fraction of output units using the convex branch
        (only used in ``mixed`` mode). Not configurable under
        ``mode="alternate"`` (must be left at ``0.5``) — the layer sets it
        internally to ``1.0`` or ``0.0`` depending on phase.
    :param init: Initializer name or :class:`~mononet.core.types.InitSpec`.
        Not configurable under ``mode="alternate"`` — the layer derives its
        own composition-aware init.
    :param bias: Whether to include a bias term (default ``True``).
    :param near_zero_scale: Private. When not ``None``, the weight is scaled
        by this factor after init and the bias is zeroed — used by
        :class:`MonoResidual` to near-zero-initialize the last layer of its
        default ``F``.
    :param prev: Only valid under ``mode="alternate"``. The preceding
        ``MonoDense`` in an alternating stack, or ``None`` for the entry
        layer. The layer's phase is the opposite of ``prev._alt_convex``
        (entry is convex), and its incoming per-coordinate mean is
        ``prev._alt_out_mean`` (``0.0`` for the entry layer). ``prev`` must
        itself be a ``mode="alternate"`` ``MonoDense``. Because Keras builds
        weights lazily, only the fan-in-independent part of the chain
        (phase, incoming mean, gain, outgoing mean) is resolved here; the
        fan-in-dependent weight std / bias fill are computed in
        :meth:`build`. ``prev`` is an init-time reference only — it is not
        retained on ``self`` and is *not* serialized by :meth:`get_config`,
        so a layer deserialized from config loses its ``alternate`` chain
        (re-chaining is a build-time concern, not a config concern).
    :raises ValueError: If ``convex_fraction != 0.5`` or ``init is not
        None`` under ``mode="alternate"``; if ``prev`` is given under a
        non-``alternate`` mode; if ``prev`` is given but is not an
        ``alternate``-mode ``MonoDense``.
    """

    def __init__(
        self,
        units: int,
        *,
        mode: Mode = "mixed",
        activation: ActivationSpec | ActivationName | None = None,
        convex_fraction: float = 0.5,
        init: InitSpec | str | None = None,
        bias: bool = True,
        near_zero_scale: float | None = None,
        prev: MonoDense | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialise MonoDense."""
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation_name = (
            "identity" if activation is None else _act_name(activation)
        )
        self.convex_fraction = convex_fraction
        self.init_name = _init_name(init)
        self._absolute_default = mode == "mixed" and init is None
        self.use_bias = bias
        self.near_zero_scale = near_zero_scale
        self._is_alternate = mode == "alternate"
        if self._is_alternate:
            if convex_fraction != 0.5:
                raise ValueError(
                    "convex_fraction is not configurable for mode='alternate'"
                )
            if init is not None:
                raise ValueError("init is not configurable for mode='alternate'")
            convex, self._alt_m_in = self._alternate_phase(prev)
            self.convex_fraction = 1.0 if convex else 0.0
            self._alt_gain, out_mean = alternating_init_params(
                self.activation_name, self._alt_m_in, convex
            )
            self._alt_convex = convex
            self._alt_out_mean = out_mean
        elif prev is not None:
            raise ValueError("prev is only valid for mode='alternate'")

    @staticmethod
    def _alternate_phase(prev: MonoDense | None) -> tuple[bool, float]:
        """Return ``(convex, m_in)`` for an alternate layer given its predecessor.

        :param prev: Preceding ``alternate``-mode ``MonoDense``, or ``None``
            for the entry layer.
        :returns: ``(convex, m_in)`` for this layer.
        :raises ValueError: If ``prev`` is given but is not an
            ``alternate``-mode ``MonoDense``.
        """
        if prev is None:
            return True, 0.0  # entry: convex, standardized input
        if getattr(prev, "mode", None) != "alternate":
            raise ValueError("prev must be an alternate-mode MonoDense")
        return (not prev._alt_convex), prev._alt_out_mean

    def build(self, input_shape: Any) -> None:
        """Create weights once the input width is known.

        :param input_shape: Shape tuple; ``input_shape[-1]`` is ``in_features``.
        """
        in_f = int(input_shape[-1])
        if self._is_alternate:
            w_std, bias_fill = alternating_weight_bias(
                self._alt_gain, self._alt_m_in, in_f
            )
            w_init = keras.initializers.RandomNormal(stddev=w_std)
            b_init = keras.initializers.Constant(bias_fill)
        elif self._absolute_default:
            gain, bias_fill = absolute_init_params(
                self.activation_name, self.convex_fraction
            )
            w_init = keras.initializers.RandomNormal(stddev=gain / math.sqrt(in_f))
            b_init = keras.initializers.Constant(bias_fill)
        else:
            w_init = self.init_name
            b_init = "zeros"
        self.w = self.add_weight(
            shape=(in_f, self.units), initializer=w_init, trainable=True, name="weight"
        )
        self.b = (
            self.add_weight(
                shape=(self.units,), initializer=b_init, trainable=True, name="bias"
            )
            if self.use_bias
            else None
        )
        if self.near_zero_scale is not None:
            self.w.assign(self.w * self.near_zero_scale)
            if self.b is not None:
                # Multiply the Variable (Variable.__mul__ returns a backend
                # tensor); ``ops.zeros_like(self.b)`` would pass the raw keras
                # Variable to the backend and fail dtype conversion under JAX.
                self.b.assign(self.b * 0.0)
        super().build(input_shape)

    def call(self, inputs: Any) -> Any:
        """Apply the monotonic dense transformation.

        :param inputs: Input tensor of shape ``(batch, in_features)``.
        :returns: Output tensor of shape ``(batch, units)``.
        """
        bias = self.b if self.b is not None else ops.zeros((self.units,))
        kernel_mode = "mixed" if self.mode == "alternate" else self.mode
        return _kernels.monotonic_dense(
            inputs,
            self.w,
            bias,
            kernel_mode,
            self.activation_name,
            self.convex_fraction,
        )

    def get_config(self) -> dict[str, Any]:
        """Serialize token/scalar fields (callables are not serializable).

        ``prev`` (an init-time-only reference; see ``__init__``) is
        deliberately not part of the config: a deserialized ``alternate``
        layer loses its chain and must be re-``prev``-chained by the caller
        if reused as a template for further layers.

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
        A custom ``F`` is not near-zero-initialised; for deep stacks initialise
        its last layer near zero (or pass ``beta_gate="scaled_elu"``) to avoid
        divergence at init.
    :param mode: Forwarded to the default ``F``. ``mixed`` (default) or
        ``split``. ``alternate`` is rejected: the default ``F`` builder can't
        thread ``prev=`` through its ``MonoDense`` layers, so pass a custom
        ``F`` built from ``prev``-chained alternate ``MonoDense`` layers
        instead.
    :param activation: Forwarded to the default ``F`` (default ``None``).
        Required when ``F`` is not provided; mutually exclusive with an
        explicit ``F``. A custom ``F`` is not serializable, so
        :meth:`get_config` emits ``activation=None`` in that case.
    :param alpha_gate: Gate token for the skip path (``shifted_elu``).
    :param beta_gate: Gate token for the dense path (``softplus``).
    :param init: Forwarded to the default ``F``.
    :param sub_depth: Number of :class:`MonoDense` layers in ``F`` (default 2;
        ``1`` = legacy single layer).  Mutually exclusive with ``F``.
    :param near_zero_scale: Scale applied to the default ``F``'s last-layer
        weight (bias zeroed) so the block starts near-identity — the deep
        default stack (``softplus`` ``beta_gate``) would otherwise diverge
        through a randomly-initialized residual branch. ``0.0`` reproduces
        exact-zero, which is not recommended: under ``mixed`` mode ``F``
        uses ``|W|``, whose gradient at ``W=0`` is ``sign(0)=0``, a fixed
        point that freezes the weights. A custom ``F`` is untouched.
    :raises ValueError: If ``F`` is ``None`` and ``activation`` is not
        provided, or if both ``F`` and ``activation`` are provided, or if
        ``mode`` is ``"alternate"``.
    """

    def __init__(
        self,
        units: int,
        *,
        F: keras.layers.Layer | None = None,  # noqa: N803
        mode: Mode = "mixed",
        activation: ActivationSpec | ActivationName | None = None,
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "softplus",
        init: InitSpec | str | None = None,
        sub_depth: int | None = None,
        near_zero_scale: float = _NEAR_ZERO_SCALE,
        **kwargs: Any,
    ) -> None:
        """Initialise MonoResidual.

        :param sub_depth: Number of :class:`MonoDense` layers inside ``F``
            (default ``None`` → 2).  ``1`` reproduces the legacy single-layer
            behaviour.  Mutually exclusive with ``F``.
        """
        super().__init__(**kwargs)
        if mode == "alternate":
            raise ValueError(
                "mode='alternate' is not supported in MonoResidual; build a custom "
                "F of alternate MonoDense layers chained with prev= instead"
            )
        self.units = units
        self.mode = mode
        self.init_name = _init_name(init)
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        if sub_depth is not None and sub_depth < 1:
            raise ValueError(f"sub_depth must be >= 1, got {sub_depth}")
        if F is not None and sub_depth is not None:
            raise ValueError("pass either F or sub_depth, not both")
        if F is None:
            if activation is None:
                raise ValueError("activation is required when F is not provided")
            self.activation_name: str | None = _act_name(activation)
            k = 2 if sub_depth is None else sub_depth
            if k == 1:
                self.F: keras.layers.Layer = MonoDense(
                    units,
                    mode=mode,
                    activation=activation,
                    init=init,
                    near_zero_scale=near_zero_scale,
                )
            else:
                sub = [
                    MonoDense(units, mode=mode, activation=activation, init=init)
                    for _ in range(k - 1)
                ]
                sub.append(
                    MonoDense(
                        units,
                        mode=mode,
                        activation=activation,
                        init=init,
                        near_zero_scale=near_zero_scale,
                    )
                )
                self.F = keras.Sequential(sub)
        else:
            if activation is not None:
                raise ValueError("pass either F or activation, not both")
            self.activation_name = None
            self.F = F

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
