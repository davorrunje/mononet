# SPDX-License-Identifier: Apache-2.0
"""Mixed-feature monotone network (JAX / Flax NNX). See risk_net_torch.py."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from mononet import MonotonicityMask
from mononet.jax import MonoInput, MonoLinear, MonoResidual


class RiskNet(nnx.Module):
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.embed1 = nnx.Linear(2, 16, rngs=rngs)
        self.embed2 = nnx.Linear(16, 8, rngs=rngs)
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.l1 = MonoLinear(11, 64, activation="elu", rngs=rngs)
        self.r1 = MonoResidual(64, 64, activation="elu", rngs=rngs)
        self.r2 = MonoResidual(64, 64, activation="elu", rngs=rngs)
        self.head = MonoLinear(64, 1, rngs=rngs)

    def __call__(self, x_mono: jnp.ndarray, x_free: jnp.ndarray) -> jnp.ndarray:
        """Combine the sign-flipped monotone features with the free embedding."""
        h = nnx.relu(self.embed1(x_free))
        h = nnx.relu(self.embed2(h))
        z = jnp.concatenate([self.mono_in(x_mono), h], axis=-1)
        return self.head(self.r2(self.r1(self.l1(z))))
