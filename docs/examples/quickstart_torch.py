# SPDX-License-Identifier: Apache-2.0
"""Quickstart: a monotone regressor in PyTorch.

Non-decreasing in every one of its 4 inputs. ``mononet`` ships layers, not
composed models — stack them with a native ``torch.nn.Sequential``.
"""

from __future__ import annotations

import torch
from torch import nn

from mononet.torch import MonoLinear, MonoResidual

model = nn.Sequential(
    MonoLinear(4, 32, activation="elu"),
    MonoResidual(32, 32, activation="elu"),
    MonoLinear(32, 1),
)

y = model(torch.rand(8, 4))
print(y.shape)  # torch.Size([8, 1]) — monotone in all 4 inputs
