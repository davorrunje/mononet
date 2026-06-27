# SPDX-License-Identifier: Apache-2.0
"""Private PyTorch kernels for monotonic primitives (stateless).

Stateless functions that take tensors and return tensors. Wrapper
classes in layers.py / models.py instantiate parameters and delegate
here.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

_SELU = torch.nn.SELU()


def activation(name: str, h: torch.Tensor) -> torch.Tensor:
    """Apply the base activation by name.

    :param name: One of ``relu``, ``elu``, ``selu``, ``softplus``.
    :param h: Input tensor.
    :returns: Activated tensor with the same shape as ``h``.
    :raises ValueError: If ``name`` is not a recognised activation.
    """
    if name == "relu":
        return torch.relu(h)
    if name == "elu":
        result: torch.Tensor = functional.elu(h)
        return result
    if name == "selu":
        selu_out: torch.Tensor = _SELU(h)
        return selu_out
    if name == "softplus":
        sp: torch.Tensor = functional.softplus(h)
        return sp
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: str, h: torch.Tensor) -> torch.Tensor:
    """Concave reflection: ``-activation(name, -h)``.

    :param name: Activation name passed to :func:`activation`.
    :param h: Input tensor.
    :returns: Concave-reflected tensor with the same shape as ``h``.
    """
    return -activation(name, -h)


def gate(token: str, raw: torch.Tensor) -> torch.Tensor:
    """Resolve and apply a gate token to a raw parameter.

    :param token: Gate type: ``shifted_elu`` or ``scaled_elu``.
    :param raw: Raw (unconstrained) gate parameter tensor.
    :returns: Gate value tensor guaranteed positive.
    :raises ValueError: If ``token`` is not a recognised gate.
    """
    if token == "shifted_elu":
        shifted: torch.Tensor = functional.elu(raw) + 1.0
        return shifted
    if token == "scaled_elu":
        eps = 1e-3
        return torch.clamp(raw, min=0.0) + eps * torch.exp(
            torch.clamp(raw, max=0.0) / eps
        )
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
) -> torch.Tensor:
    """PyTorch monotonic dense kernel (see core.reference.monotonic_dense).

    :param x: Input tensor of shape ``(batch, in_features)``.
    :param weights: Weight matrix of shape ``(in_features, units)``.
    :param bias: Bias vector of shape ``(units,)``.
    :param mode: ``switch`` or ``absolute``.
    :param activation_name: Base activation name.
    :param convex_fraction: Fraction of units with convex activation (absolute mode).
    :returns: Output tensor of shape ``(batch, units)``.
    :raises ValueError: If ``mode`` is not ``switch`` or ``absolute``.
    """
    if mode == "switch":
        w_pos = torch.clamp(weights, min=0.0)
        w_neg = torch.clamp(weights, max=0.0)
        return activation(activation_name, x @ w_pos + bias) - activation(
            activation_name, x @ w_neg + bias
        )
    if mode == "absolute":
        h = x @ torch.abs(weights) + bias
        m = weights.shape[1]
        c = math.ceil(convex_fraction * m)
        left = activation(activation_name, h[:, :c])
        right = concave_reflection(activation_name, h[:, c:])
        return torch.cat([left, right], dim=1)
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch dual-gated monotone residual kernel.

    :param x: Input tensor of shape ``(batch, in_features)``.
    :param weights: Weight matrix of shape ``(in_features, units)``.
    :param bias: Bias vector of shape ``(units,)``.
    :param alpha: Raw skip-gate parameter.
    :param beta: Raw transform-gate parameter.
    :param mode: Passed to :func:`monotonic_dense`.
    :param activation_name: Base activation name.
    :param convex_fraction: Fraction of convex units (absolute mode).
    :param alpha_gate: Gate token for the skip path.
    :param beta_gate: Gate token for the transform path.
    :param skip_weight: Optional log-scale skip weight; if given, skip is
        ``x @ exp(skip_weight)`` instead of ``x``.
    :returns: Output tensor.
    """
    f = monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction)
    skip = x if skip_weight is None else x @ torch.exp(skip_weight)
    return gate(alpha_gate, alpha) * skip + gate(beta_gate, beta) * f
