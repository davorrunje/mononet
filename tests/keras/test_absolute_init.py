# SPDX-License-Identifier: Apache-2.0
import math
import os
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
pytest.importorskip("keras")

from mononet.core.init import absolute_init_params
from mononet.keras import MonoDense


def _build(units: int, in_f: int, **kw: Any) -> MonoDense:
    layer = MonoDense(units, **kw)
    layer.build((None, in_f))
    return layer


def test_absolute_default_weight_scale_and_bias() -> None:
    in_f, units = 256, 512
    layer = _build(units, in_f, mode="absolute", activation="elu")
    gain, bias = absolute_init_params("elu", 0.5)
    got = float(np.std(np.asarray(layer.w)))
    assert abs(got - gain / math.sqrt(in_f)) < 0.05 * gain / math.sqrt(in_f)
    assert np.allclose(np.asarray(layer.b), np.full((units,), bias))


def test_absolute_bias_nonzero_off_half() -> None:
    layer = _build(64, 64, mode="absolute", activation="elu", convex_fraction=0.25)
    _, bias = absolute_init_params("elu", 0.25)
    assert bias != 0.0
    assert np.allclose(np.asarray(layer.b), np.full((64,), bias))


def test_switch_default_unchanged() -> None:
    layer = _build(64, 64, mode="switch", activation="elu")
    assert np.allclose(np.asarray(layer.b), np.zeros((64,)))
