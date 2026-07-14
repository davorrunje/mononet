# SPDX-License-Identifier: Apache-2.0
"""Mixed-feature monotone network (PyTorch).

Monotone in 3 features (2 non-decreasing, 1 non-increasing) via ``MonoInput``,
and unconstrained in 2 non-monotone features, which are embedded through a
plain MLP. The embedding absorbs the non-monotonicity, so the composite map is
monotone in ``x_mono`` and free in ``x_free``. Mixed mode is the default.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mononet import MonotonicityMask
from mononet.torch import MonoInput, MonoLinear, MonoResidual


class RiskNet(nn.Module):
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.net = nn.Sequential(
            MonoLinear(11, 64, activation="elu"),
            MonoResidual(64, 64, activation="elu"),
            MonoResidual(64, 64, activation="elu"),
            MonoLinear(64, 1),
        )

    def forward(self, x_mono: torch.Tensor, x_free: torch.Tensor) -> torch.Tensor:
        """Combine the sign-flipped monotone features with the free embedding."""
        z = torch.cat([self.mono_in(x_mono), self.embed(x_free)], dim=-1)
        return self.net(z)
