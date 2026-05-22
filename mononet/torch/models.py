"""PyTorch monotonic-model compositions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoMLP(nn.Module):
    """Multi-layer monotonic MLP, PyTorch backend."""

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
    ) -> None:
        """Initialise MonoMLP (not yet implemented)."""
        super().__init__()
        raise NotImplementedError(
            "MonoMLP PyTorch composition lands in the follow-up plan."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """Apply the monotonic MLP."""
        raise NotImplementedError
