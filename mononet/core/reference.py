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

    from mononet.core.types import ActivationName, ActivationSpec, MonotonicityMask

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
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
    """Single-layer monotonic transformation (NumPy reference).

    :param x: Input array of shape `(batch, in_features)`.
    :param weights: Unconstrained weights of shape
        `(in_features, out_features)`.
    :param bias: Bias vector of shape `(out_features,)`.
    :param mask: Per-input monotonicity mask.
    :param activation: Activation specification.
    :returns: Output array of shape `(batch, out_features)`.
    """
    raise NotImplementedError(
        "monotonic_dense reference implementation lands in the follow-up plan."
    )


def monotonic_mlp(
    x: npt.NDArray[np.floating],
    weights: list[npt.NDArray[np.floating]],
    biases: list[npt.NDArray[np.floating]],
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
    """Multi-layer monotonic MLP (NumPy reference).

    :param x: Input array of shape `(batch, in_features)`.
    :param weights: Per-layer weight arrays.
    :param biases: Per-layer bias vectors.
    :param mask: Monotonicity mask applied to the first layer.
    :param activation: Activation used between hidden layers.
    :returns: Output array of shape `(batch, weights[-1].shape[1])`.
    """
    raise NotImplementedError(
        "monotonic_mlp reference implementation lands in the follow-up plan."
    )
