"""Private PyTorch kernels for monotonic primitives.

Stateless functions that take tensors and return tensors. Wrapper
classes in layers.py / models.py instantiate parameters and delegate
here. Real implementations land in the follow-up algorithm plan.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> torch.Tensor:
    """PyTorch kernel for the monotonic dense transformation."""
    raise NotImplementedError(
        "monotonic_dense PyTorch kernel lands in the follow-up plan."
    )
