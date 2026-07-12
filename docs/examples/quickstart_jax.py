# SPDX-License-Identifier: Apache-2.0
"""Quickstart: a monotone regressor in JAX / Flax NNX.

Non-decreasing in every one of its 4 inputs. Dense layers take an explicit
``rngs`` for weight initialization.
"""

from __future__ import annotations

import jax
from flax import nnx

from mononet.jax import MonoLinear, MonoResidual

rngs = nnx.Rngs(0)
model = nnx.Sequential(
    MonoLinear(4, 32, activation="elu", rngs=rngs),
    MonoResidual(32, 32, activation="elu", rngs=rngs),
    MonoLinear(32, 1, rngs=rngs),
)

y = model(jax.random.uniform(jax.random.key(0), (8, 4)))
print(y.shape)  # (8, 1) — monotone in all 4 inputs
