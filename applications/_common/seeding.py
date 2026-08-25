# SPDX-License-Identifier: Apache-2.0
"""Deterministic RNG helpers shared across applications.

All randomness in the applications area flows through these helpers so that
point sampling and observation generation are reproducible and identical
across backends.
"""

from __future__ import annotations

import numpy as np


def rng(seed: int) -> np.random.Generator:
    """Return a NumPy generator seeded deterministically.

    :param seed: Non-negative integer seed.
    :returns: A `numpy.random.Generator` (PCG64) seeded with `seed`.
    """
    return np.random.default_rng(seed)


def split_seeds(seed: int, n: int) -> list[int]:
    """Derive `n` independent integer seeds from a base `seed`.

    Uses `numpy.random.SeedSequence` so the child seeds are well separated and
    reproducible.

    :param seed: Base seed.
    :param n: Number of child seeds to produce.
    :returns: A list of `n` deterministic non-negative integer seeds.
    """
    state = np.random.SeedSequence(seed).generate_state(n)
    return [int(s) for s in state]
