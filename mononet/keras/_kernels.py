"""Private Keras (keras.ops) kernels for monotonic primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: Any,
    weights: Any,
    bias: Any,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> Any:
    """Keras kernel for the monotonic dense transformation.

    Uses keras.ops, so it works with the JAX, TensorFlow, or PyTorch
    Keras backend.
    """
    raise NotImplementedError(
        "monotonic_dense Keras kernel lands in the follow-up plan."
    )
