# SPDX-License-Identifier: Apache-2.0
"""Behavior, deprecation warning, and serialization for legacy MonoDense."""

from __future__ import annotations

import contextlib
import os
import warnings

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense  # noqa: E402


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
