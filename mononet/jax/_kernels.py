# SPDX-License-Identifier: Apache-2.0
"""Private JAX kernels for monotonic primitives (pure-functional)."""

from __future__ import annotations

import math

import jax.nn as jnn
import jax.numpy as jnp


def activation(name: str, h: jnp.ndarray) -> jnp.ndarray:
    """Apply the base activation by name.

    :param name: One of ``relu``, ``elu``, ``selu``, ``softplus``.
    :param h: Input array.
    :returns: Element-wise activation applied to ``h``.
    :raises ValueError: If ``name`` is not a supported activation.
    """
    if name == "relu":
        return jnn.relu(h)
    if name == "elu":
        return jnn.elu(h)
    if name == "selu":
        return jnn.selu(h)
    if name == "softplus":
        return jnn.softplus(h)
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: str, h: jnp.ndarray) -> jnp.ndarray:
    """Concave reflection defined as ``-activation(-h)``.

    :param name: Activation name passed to :func:`activation`.
    :param h: Input array.
    :returns: ``-activation(name, -h)``.
    """
    return -activation(name, -h)


def gate(token: str, raw: jnp.ndarray) -> jnp.ndarray:
    """Resolve and apply a gate token to a raw parameter.

    :param token: Either ``shifted_elu`` or ``scaled_elu``.
    :param raw: Raw (unconstrained) gate parameter array.
    :returns: Strictly-positive gated value.
    :raises ValueError: If ``token`` is not a supported gate.
    """
    if token == "shifted_elu":
        return jnn.elu(raw) + 1.0
    if token == "scaled_elu":
        eps = 1e-3
        return jnp.maximum(raw, 0.0) + eps * jnp.exp(jnp.minimum(raw, 0.0) / eps)
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
) -> jnp.ndarray:
    """JAX monotonic dense kernel (pure-functional).

    :param x: Input tensor of shape ``(batch, in_features)``.
    :param weights: Weight matrix of shape ``(in_features, units)``.
    :param bias: Bias vector of shape ``(units,)``.
    :param mode: ``switch`` or ``absolute``.
    :param activation_name: Base activation name.
    :param convex_fraction: Fraction of output units that are convex
        (used only when ``mode="absolute"``).
    :returns: Output tensor of shape ``(batch, units)``.
    :raises ValueError: If ``mode`` is not ``switch`` or ``absolute``.
    """
    if mode == "switch":
        w_pos = jnp.maximum(weights, 0.0)
        w_neg = jnp.minimum(weights, 0.0)
        return activation(activation_name, x @ w_pos + bias) - activation(
            activation_name, x @ w_neg + bias
        )
    if mode == "absolute":
        h = x @ jnp.abs(weights) + bias
        c = math.ceil(convex_fraction * weights.shape[1])
        left = activation(activation_name, h[:, :c])
        right = concave_reflection(activation_name, h[:, c:])
        return jnp.concatenate([left, right], axis=1)
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """JAX dual-gated monotone residual kernel (pure-functional).

    :param x: Input tensor of shape ``(batch, features)``.
    :param weights: Weight matrix for the dense sublayer.
    :param bias: Bias vector for the dense sublayer.
    :param alpha: Raw skip-path gate parameter.
    :param beta: Raw dense-path gate parameter.
    :param mode: ``switch`` or ``absolute`` passed to :func:`monotonic_dense`.
    :param activation_name: Base activation name.
    :param convex_fraction: Convex fraction for ``absolute`` mode.
    :param alpha_gate: Gate token for the skip path.
    :param beta_gate: Gate token for the dense path.
    :param skip_weight: Optional log-scale projection matrix; when ``None``
        the identity skip ``x`` is used.
    :returns: Residual output of the same shape as ``x``.
    """
    f = monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction)
    skip = x if skip_weight is None else x @ jnp.exp(skip_weight)
    return gate(alpha_gate, alpha) * skip + gate(beta_gate, beta) * f
