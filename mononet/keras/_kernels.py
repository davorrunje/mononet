# SPDX-License-Identifier: Apache-2.0
"""Private Keras 3 kernels for monotonic primitives (keras.ops only)."""

from __future__ import annotations

import math
from typing import Any

from keras import ops


def activation(name: str, h: Any) -> Any:
    """Apply the base activation by name.

    :param name: One of ``relu``, ``elu``, ``selu``, ``softplus``.
    :param h: Input tensor (any keras-compatible array).
    :returns: Activated tensor with the same shape as `h`.
    :raises ValueError: If `name` is not a supported activation.
    """
    if name == "relu":
        return ops.relu(h)
    if name == "elu":
        return ops.elu(h)
    if name == "selu":
        return ops.selu(h)
    if name == "softplus":
        return ops.softplus(h)
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: str, h: Any) -> Any:
    """Concave reflection defined as ``-activation(-h)``.

    :param name: Activation name passed to :func:`activation`.
    :param h: Input tensor.
    :returns: Negated activation of negated input, same shape as `h`.
    """
    return -activation(name, -h)


def gate(token: str, raw: Any) -> Any:
    """Resolve and apply a gate token to a raw (unconstrained) parameter.

    :param token: One of ``shifted_elu`` or ``scaled_elu``.
    :param raw: Raw (unconstrained) gate parameter tensor.
    :returns: Strictly-positive gate value, same shape as `raw`.
    :raises ValueError: If `token` is not a supported gate token.
    """
    if token == "shifted_elu":
        return ops.elu(raw) + 1.0
    if token == "scaled_elu":
        eps = 1e-3
        return ops.maximum(raw, 0.0) + eps * ops.exp(ops.minimum(raw, 0.0) / eps)
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: Any,
    weights: Any,
    bias: Any,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
) -> Any:
    """Keras monotonic dense kernel (backend-agnostic via ``keras.ops``).

    :param x: Input tensor of shape ``(batch, in_features)``.
    :param weights: Weight matrix of shape ``(in_features, units)``.
    :param bias: Bias vector of shape ``(units,)``.
    :param mode: Either ``switch`` or ``absolute``.
    :param activation_name: Base activation name (``relu``, ``elu``, ``selu``,
        ``softplus``).
    :param convex_fraction: Fraction of output units that use the convex
        activation; remainder use the concave reflection. Only used for
        ``absolute`` mode.
    :returns: Output tensor of shape ``(batch, units)``.
    :raises ValueError: If `mode` is not ``switch`` or ``absolute``.
    """
    if mode == "switch":
        w_pos = ops.maximum(weights, 0.0)
        w_neg = ops.minimum(weights, 0.0)
        return activation(activation_name, ops.matmul(x, w_pos) + bias) - activation(
            activation_name, ops.matmul(x, w_neg) + bias
        )
    if mode == "absolute":
        h = ops.matmul(x, ops.abs(weights)) + bias
        c = math.ceil(convex_fraction * int(weights.shape[1]))
        left = activation(activation_name, h[:, :c])
        right = concave_reflection(activation_name, h[:, c:])
        return ops.concatenate([left, right], axis=1)
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: Any,
    weights: Any,
    bias: Any,
    alpha: Any,
    beta: Any,
    *,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: Any | None = None,
) -> Any:
    """Keras dual-gated monotone residual kernel.

    :param x: Input tensor of shape ``(batch, in_features)``.
    :param weights: Weight matrix of shape ``(in_features, units)``.
    :param bias: Bias vector of shape ``(units,)``.
    :param alpha: Unconstrained skip-gate parameter (scalar or broadcastable).
    :param beta: Unconstrained dense-gate parameter (scalar or broadcastable).
    :param mode: Either ``switch`` or ``absolute``; forwarded to
        :func:`monotonic_dense`.
    :param activation_name: Base activation name; forwarded to
        :func:`monotonic_dense`.
    :param convex_fraction: Convex fraction for ``absolute`` mode.
    :param alpha_gate: Gate token for the skip path (default ``shifted_elu``).
    :param beta_gate: Gate token for the dense path (default ``scaled_elu``).
    :param skip_weight: Optional projection log-weight matrix of shape
        ``(in_features, units)``; when provided the skip path is
        ``x @ exp(skip_weight)``.
    :returns: Output tensor of shape ``(batch, units)``.
    """
    f = monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction)
    skip = x if skip_weight is None else ops.matmul(x, ops.exp(skip_weight))
    return gate(alpha_gate, alpha) * skip + gate(beta_gate, beta) * f
