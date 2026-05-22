"""PyTorch idiomatic layer wrappers around mononet kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoLinear(nn.Module):
    """Monotonic analogue of `torch.nn.Linear`.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        monotonicity: Per-input-feature monotonicity mask.
        activation: Activation specification (resolved by the kernel).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
    ) -> None:
        """Initialise MonoLinear (not yet implemented)."""
        super().__init__()
        raise NotImplementedError(
            "MonoLinear PyTorch wrapper lands in the follow-up plan."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Apply the monotonic linear transformation."""
        raise NotImplementedError
