# SPDX-License-Identifier: Apache-2.0
"""Structure and forward-pass tests for legacy network builders."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense, create_type_1, create_type_2  # noqa: E402


def test_create_type_1_builds_runnable_model() -> None:
    legacy._WARNED = True
    inp = keras.Input(shape=(4,))
    out = create_type_1(
        inp,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=3,
        monotonicity_indicator=[1, 1, -1, 0],
    )
    model = keras.Model(inp, out)
    y = model(np.zeros((2, 4), dtype="float32"))
    assert tuple(y.shape) == (2, 1)


def test_create_type_1_classmethod_matches_function() -> None:
    legacy._WARNED = True
    inp = keras.Input(shape=(3,))
    out = MonoDense.create_type_1(
        inp, units=4, final_units=2, activation="relu", n_layers=2
    )
    model = keras.Model(inp, out)
    assert tuple(model(np.zeros((1, 3), dtype="float32")).shape) == (1, 2)


def test_create_type_2_builds_runnable_model() -> None:
    legacy._WARNED = True
    inp = keras.Input(shape=(4,))
    out = create_type_2(
        inp,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=2,
        monotonicity_indicator=[1, -1, 0, 1],
    )
    model = keras.Model(inp, out)
    assert tuple(model(np.zeros((2, 4), dtype="float32")).shape) == (2, 1)
