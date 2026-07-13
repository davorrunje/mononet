# SPDX-License-Identifier: Apache-2.0
"""Behavior, deprecation warning, and serialization for legacy MonoDense."""

from __future__ import annotations

import contextlib
import json
import os
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense, create_type_1  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


@contextlib.contextmanager
def warnings_none():  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


def test_construction_emits_deprecation_warning() -> None:
    legacy._WARNED = False
    with pytest.warns(DeprecationWarning, match="mononet.legacy"):
        MonoDense(4)


def test_warning_fires_only_once() -> None:
    legacy._WARNED = False
    with pytest.warns(DeprecationWarning, match="mononet.legacy"):
        MonoDense(4)
    with warnings_none():
        MonoDense(4)  # second construction: no warning


def test_rejects_convex_and_concave() -> None:
    legacy._WARNED = True
    with pytest.raises(ValueError, match="convex and concave"):
        MonoDense(4, is_convex=True, is_concave=True)


def test_rejects_bad_activation_weights() -> None:
    legacy._WARNED = True
    with pytest.raises(ValueError, match="activation_weights"):
        MonoDense(4, activation_weights=(1.0, 1.0))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="activation_weights"):
        MonoDense(4, activation_weights=(-1.0, 1.0, 1.0))


def test_forward_pass_is_increasing_for_positive_indicator() -> None:
    legacy._WARNED = True
    layer = MonoDense(1, activation="relu", monotonicity_indicator=1)
    layer.build((None, 3))
    x0 = ops.convert_to_tensor(np.zeros((1, 3), dtype="float32"))
    x1 = ops.convert_to_tensor(np.ones((1, 3), dtype="float32"))
    y0 = float(np.asarray(layer(x0))[0, 0])
    y1 = float(np.asarray(layer(x1))[0, 0])
    assert y1 >= y0  # non-decreasing along an all-increasing input


def test_get_config_round_trip() -> None:
    legacy._WARNED = True
    layer = MonoDense(
        5,
        activation="elu",
        monotonicity_indicator=1,
        is_convex=True,
        activation_weights=(2.0, 3.0, 1.0),
    )
    cfg = layer.get_config()
    assert cfg["units"] == 5
    assert cfg["activation"] == "elu"
    assert cfg["is_convex"] is True
    assert cfg["activation_weights"] == (2.0, 3.0, 1.0)
    rebuilt = MonoDense.from_config(cfg)
    assert rebuilt.units == 5


def test_get_config_round_trip_after_build_is_json_serializable() -> None:
    """Regression test for build() overwriting monotonicity_indicator.

    ``build()`` replaces ``monotonicity_indicator`` with an ``np.ndarray``,
    which ``get_config()`` must convert back to a JSON-serializable form (a
    plain list) for ``json.dumps()`` / ``keras.models.save_model()`` to work.
    """
    legacy._WARNED = True
    layer = MonoDense(3, activation="relu", monotonicity_indicator=[1, -1, 0])
    layer.build((None, 3))
    assert isinstance(layer.monotonicity_indicator, np.ndarray)

    cfg = layer.get_config()
    # Must not raise TypeError: Object of type ndarray is not JSON serializable.
    serialized = json.dumps(cfg)
    assert isinstance(cfg["monotonicity_indicator"], list)

    deserialized = json.loads(serialized)
    rebuilt = MonoDense.from_config(deserialized)
    assert rebuilt.units == 3
    rebuilt.build((None, 3))
    x = ops.convert_to_tensor(np.ones((1, 3), dtype="float32"))
    y = np.asarray(rebuilt(x))
    assert y.shape == (1, 3)


@pytest.mark.filterwarnings(
    "ignore:__array__ implementation doesn't accept a copy keyword:DeprecationWarning"
)
def test_save_model_on_built_mono_dense_model(tmp_path: Path) -> None:
    """End-to-end guard that a built MonoDense model is save_model-able.

    Every layer's ``get_config()`` must be JSON-serializable.
    ``save_model`` internally calls ``Variable.numpy()``, which under the JAX
    backend + NumPy 2.x raises an unrelated DeprecationWarning from inside
    Keras itself (see the comment in ``test_forward_pass_with_use_bias_false``
    below for the same root cause); it is filtered here rather than sidestepped
    since ``save_model`` itself is the thing under test.
    """
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(3)]
    out = create_type_1(
        inputs,
        units=4,
        final_units=1,
        activation="relu",
        n_layers=2,
        monotonicity_indicator=[1, -1, 0],
    )
    model = keras.Model(inputs, out)
    # Force a build/call so any MonoDense layers replace their indicator with
    # an np.ndarray (build() happens lazily via the functional call above,
    # but exercise the same path other tests use for clarity).
    xs = [np.zeros((2, 1), dtype="float32") for _ in range(3)]
    model(xs)

    for layer in model.layers:
        json.dumps(layer.get_config())

    keras.models.save_model(model, str(tmp_path / "m.keras"))


def test_forward_pass_with_use_bias_false() -> None:
    legacy._WARNED = True
    layer = MonoDense(
        1,
        activation="relu",
        monotonicity_indicator=1,
        use_bias=False,
        is_convex=True,
    )
    layer.build((None, 2))

    # Assert no bias weight exists (only kernel, no bias). Avoid
    # layer.get_weights() here: it calls Variable.numpy() -> np.array(self),
    # which under the JAX backend + NumPy 2.x raises a DeprecationWarning from
    # inside Keras itself (Variable.__array__ doesn't accept `copy`) — beyond
    # our control, so we sidestep it rather than trigger it.
    assert len(layer.weights) == 1

    # Set a known kernel
    kernel = np.array([[2.0], [3.0]], dtype="float32")
    layer.set_weights([kernel])

    # Input: [1, 1]
    # Pre-activation: matmul([1, 1], [[2], [3]]) = 1*2 + 1*3 = 5
    # Post-activation (relu with is_convex=True): relu(5) = 5
    x = ops.convert_to_tensor(np.array([[1.0, 1.0]], dtype="float32"))
    expected = np.array([[5.0]], dtype="float32")

    output = layer(x)
    assert np.allclose(np.asarray(output), expected, atol=1e-5)
