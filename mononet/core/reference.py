"""NumPy reference implementations of the monotonic primitives.

These are the **arithmetic ground truth**: every backend kernel is
asserted equivalent to these functions within a fixed tolerance
(see tests/equivalence/). Real implementations land in a follow-up plan;
this module currently raises NotImplementedError but locks down the
function signatures.

Reference paper: https://arxiv.org/abs/2205.11775
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
    """Single-layer monotonic transformation (NumPy reference).

    Args:
        x: Input array of shape (batch, in_features).
        weights: Unconstrained weights of shape (in_features, out_features).
        bias: Bias vector of shape (out_features,).
        mask: Per-input monotonicity mask.
        activation: Activation specification.

    Returns:
        Output array of shape (batch, out_features).
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

    Args:
        x: Input array of shape (batch, in_features).
        weights: Per-layer weight arrays.
        biases: Per-layer bias vectors.
        mask: Monotonicity mask applied to the first layer.
        activation: Activation used between hidden layers.

    Returns:
        Output array of shape (batch, weights[-1].shape[1]).
    """
    raise NotImplementedError(
        "monotonic_mlp reference implementation lands in the follow-up plan."
    )
