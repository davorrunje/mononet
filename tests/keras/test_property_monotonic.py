# SPDX-License-Identifier: Apache-2.0
"""Property test: Keras layers are non-decreasing in every input."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from typing import TYPE_CHECKING  # noqa: E402

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from keras import ops  # noqa: E402

from mononet.keras import MonoDense  # noqa: E402

if TYPE_CHECKING:
    from mononet.core.config import Mode


@settings(deadline=None, max_examples=30)
@given(mode=st.sampled_from(["split", "mixed"]), seed=st.integers(0, 10_000))
def test_mono_dense_nondecreasing(mode: Mode, seed: int) -> None:
    rng = np.random.default_rng(seed)
    layer = MonoDense(4, mode=mode)
    x = ops.convert_to_tensor(rng.normal(size=(6, 3)))
    bump = np.zeros((6, 3))
    bump[:, 1] = 0.5
    delta = layer(x + ops.convert_to_tensor(bump)) - layer(x)
    assert bool(ops.all(delta >= -1e-5))
