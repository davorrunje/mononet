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


# NOTE: the original API's per-feature `monotonicity_indicator` list is only
# supported with a list/dict of per-feature single-column Input tensors (its
# actual upstream usage) — NOT a single multi-feature tensor. The builders are
# a faithful port, so tests must use that same calling convention. A single
# tensor is only valid with a scalar indicator (int), which the classmethod
# test below exercises.
def test_create_type_1_builds_runnable_model() -> None:
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(4)]
    out = create_type_1(
        inputs,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=3,
        monotonicity_indicator=[1, 1, -1, 0],
    )
    model = keras.Model(inputs, out)
    xs = [np.zeros((2, 1), dtype="float32") for _ in range(4)]
    assert tuple(model(xs).shape) == (2, 1)


def test_create_type_1_accepts_dict_inputs() -> None:
    legacy._WARNED = True
    inputs = {
        "a": keras.Input(shape=(1,), name="a"),
        "b": keras.Input(shape=(1,), name="b"),
    }
    out = create_type_1(
        inputs,
        units=4,
        final_units=1,
        activation="elu",
        n_layers=2,
        monotonicity_indicator={"a": 1, "b": -1},
    )
    model = keras.Model(inputs, out)
    y = model(
        {
            "a": np.zeros((2, 1), dtype="float32"),
            "b": np.zeros((2, 1), dtype="float32"),
        }
    )
    assert tuple(y.shape) == (2, 1)


def test_create_type_1_classmethod_matches_function() -> None:
    legacy._WARNED = True
    # single tensor + scalar (int) indicator is the one single-tensor case the
    # original supports; default monotonicity_indicator=1.
    inp = keras.Input(shape=(3,))
    out = MonoDense.create_type_1(
        inp, units=4, final_units=2, activation="relu", n_layers=2
    )
    model = keras.Model(inp, out)
    assert tuple(model(np.zeros((1, 3), dtype="float32")).shape) == (1, 2)


def test_create_type_2_builds_runnable_model() -> None:
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(4)]
    out = create_type_2(
        inputs,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=2,
        monotonicity_indicator=[1, -1, 0, 1],
    )
    model = keras.Model(inputs, out)
    xs = [np.zeros((2, 1), dtype="float32") for _ in range(4)]
    assert tuple(model(xs).shape) == (2, 1)


def test_create_type_2_all_increasing_hits_inherited_name_collision() -> None:
    """Document an inherited upstream limitation (see ``create_type_2`` docstring).

    An all-increasing per-feature ``monotonicity_indicator`` with enough
    hidden layers makes a per-feature layer name collide with a shared-block
    layer name (both land on ``mono_dense_1_increasing`` and
    ``mono_dense_2_increasing`` here). This matches the original
    ``airtai/monotonic-nn`` behavior and is intentionally not fixed, so this
    test pins the current (broken) behavior rather than a desired one.
    """
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(3)]
    out = create_type_2(
        inputs,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=3,
        monotonicity_indicator=[1, 1, 1],
    )
    with pytest.raises(ValueError, match="used 2 times"):
        keras.Model(inputs, out)
