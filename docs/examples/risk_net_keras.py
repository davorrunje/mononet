# SPDX-License-Identifier: Apache-2.0
"""Mixed-feature monotone network (Keras 3). See risk_net_torch.py."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np

from mononet import MonotonicityMask
from mononet.keras import MonoDense, MonoInput, MonoResidual


class RiskNet(keras.Model):  # type: ignore[misc]
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed1 = keras.layers.Dense(16, activation="relu")
        self.embed2 = keras.layers.Dense(8, activation="relu")
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.l1 = MonoDense(64, activation="elu")
        self.r1 = MonoResidual(64, activation="elu")
        self.r2 = MonoResidual(64, activation="elu")
        self.head = MonoDense(1)

    def call(self, x_mono: Any, x_free: Any) -> Any:
        """Combine the sign-flipped monotone features with the free embedding."""
        h = self.embed2(self.embed1(x_free))
        z = keras.ops.concatenate([self.mono_in(x_mono), h], axis=-1)
        return self.head(self.r2(self.r1(self.l1(z))))
