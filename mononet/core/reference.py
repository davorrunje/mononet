"""NumPy reference implementations of the monotonic primitives.

Arithmetic ground truth for the cross-backend equivalence harness. Papers:
https://arxiv.org/abs/2205.11775 (absolute mode) and
https://arxiv.org/abs/2505.02537 (switch mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from mononet.core.types import ActivationName, ActivationSpec

_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805
_GATE_EPS = 1e-3


def base_activation(
    name: ActivationName, x: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Apply base activation `rho_breve in A_breve` element-wise.

    :param name: One of `relu`, `elu`, `selu`, `softplus`.
    :param x: Input array.
    :returns: `rho_breve(x)`.
    :raises ValueError: If `name` is not a known activation.
    """
    if name == "relu":
        return np.maximum(x, 0.0)  # type: ignore[no-any-return]
    if name == "elu":
        return np.where(x > 0.0, x, np.expm1(x))
    if name == "selu":
        return _SELU_SCALE * np.where(x > 0.0, x, _SELU_ALPHA * np.expm1(x))
    if name == "softplus":
        return np.logaddexp(0.0, x)  # type: ignore[no-any-return]
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(
    name: ActivationName, x: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute the concave point reflection `rho_hat(x) = -rho_breve(-x)`.

    :param name: Base activation name.
    :param x: Input array.
    :returns: `rho_hat(x)`.
    """
    return -base_activation(name, -x)


def apply_gate(token: str, raw: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Resolve a gate string token and apply it to a raw parameter.

    :param token: `shifted_elu` (value 1, derivative 1 at 0) or `scaled_elu`
        (value `ε`, derivative 1 at 0).
    :param raw: Raw learnable gate parameter.
    :returns: A strictly-positive gate value.
    :raises ValueError: If the token is unknown.
    """
    if token == "shifted_elu":
        return np.where(raw > 0.0, raw, np.expm1(raw)) + 1.0
    if token == "scaled_elu":
        return np.maximum(raw, 0.0) + _GATE_EPS * np.exp(  # type: ignore[no-any-return]
            np.minimum(raw, 0.0) / _GATE_EPS
        )
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    mode: str,
    activation: ActivationSpec,
    convex_fraction: float = 0.5,
) -> npt.NDArray[np.floating]:
    """Single monotonic dense transformation (NumPy reference).

    Non-decreasing in every input. `switch` uses the post-activation switch
    `rho(W_pos @ x + b) - rho(W_neg @ x + b)`; `absolute` uses `|W| @ x + b`
    with the first `ceil(convex_fraction * m)` neurons convex and the rest
    concave.

    :param x: Input array of shape `(batch, in_features)`.
    :param weights: Weights of shape `(in_features, out_features)`.
    :param bias: Bias of shape `(out_features,)`.
    :param mode: `"switch"` or `"absolute"`.
    :param activation: Base activation rho_breve.
    :param convex_fraction: Convex-neuron fraction (absolute mode only).
    :returns: Output array of shape `(batch, out_features)`.
    :raises ValueError: If `mode` is not recognised.
    """
    name = activation.name
    if mode == "switch":
        w_pos = np.maximum(weights, 0.0)
        w_neg = np.minimum(weights, 0.0)
        return base_activation(name, x @ w_pos + bias) - base_activation(
            name, x @ w_neg + bias
        )
    if mode == "absolute":
        h = x @ np.abs(weights) + bias
        m = weights.shape[1]
        c = int(np.ceil(convex_fraction * m))
        out: npt.NDArray[np.floating] = np.empty_like(h)
        out[:, :c] = base_activation(name, h[:, :c])
        out[:, c:] = concave_reflection(name, h[:, c:])
        return out
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    alpha: npt.NDArray[np.floating],
    beta: npt.NDArray[np.floating],
    *,
    mode: str = "switch",
    activation: ActivationSpec,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Dual-gated monotone residual block (NumPy reference).

    `y = g_alpha(alpha)*skip(x) + g_beta(beta)*F(x)`, with `F` a
    `monotonic_dense` and `skip` the identity (or `exp(skip_weight)`
    projection).

    :param x: Input `(batch, in_features)`.
    :param weights: `F` weights `(in_features, units)`.
    :param bias: `F` bias `(units,)`.
    :param alpha: Scalar raw skip-gate parameter.
    :param beta: Scalar raw residual-gate parameter.
    :param mode: `F` mode.
    :param activation: `F` base activation.
    :param convex_fraction: `F` convex fraction (absolute mode).
    :param alpha_gate: Skip-gate token.
    :param beta_gate: Residual-gate token.
    :param skip_weight: Projection weights `(in_features, units)`, or `None`
        for an identity skip (requires `in_features == units`).
    :returns: Output `(batch, units)`.
    """
    f = monotonic_dense(x, weights, bias, mode, activation, convex_fraction)
    skip = x if skip_weight is None else x @ np.exp(skip_weight)
    return apply_gate(alpha_gate, alpha) * skip + apply_gate(beta_gate, beta) * f
