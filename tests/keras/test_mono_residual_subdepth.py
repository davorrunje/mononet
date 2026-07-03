import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

pytest.importorskip("keras")

import keras
from keras import ops

from mononet.keras import MonoDense, MonoResidual


def test_default_builds_two_monodense() -> None:
    layer = MonoResidual(8, mode="absolute", activation="elu")  # default sub_depth -> 2
    assert isinstance(layer.F, keras.Sequential)
    assert sum(isinstance(m, MonoDense) for m in layer.F.layers) == 2


def test_subdepth_builds_k_monodense() -> None:
    layer = MonoResidual(8, mode="absolute", activation="elu", sub_depth=3)
    assert isinstance(layer.F, keras.Sequential)
    assert sum(isinstance(m, MonoDense) for m in layer.F.layers) == 3


def test_subdepth1_is_single_monodense() -> None:
    layer = MonoResidual(8, mode="absolute", sub_depth=1)
    assert isinstance(layer.F, MonoDense)


def test_f_alone_is_used() -> None:
    f = MonoDense(8, mode="absolute")
    layer = MonoResidual(8, F=f)
    assert layer.F is f


def test_f_and_explicit_subdepth_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, F=MonoDense(8, mode="absolute"), sub_depth=2)


def test_subdepth_below_one_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, mode="absolute", sub_depth=0)


def _nondecreasing(units: int, in_f: int, mode: str) -> None:
    layer = MonoResidual(units, mode=mode, activation="elu", sub_depth=2)
    x = ops.convert_to_tensor(
        np.random.default_rng(1).standard_normal((64, in_f)).astype("float32")
    )
    layer(x)  # build
    layer.beta.assign(ops.convert_to_tensor(0.7, dtype=layer.beta.dtype))
    y0 = ops.convert_to_numpy(layer(x))
    for i in range(in_f):
        xp = np.array(ops.convert_to_numpy(x))
        xp[:, i] += 0.5
        y1 = ops.convert_to_numpy(layer(ops.convert_to_tensor(xp)))
        assert float((y1 - y0).min()) >= -1e-3


def test_monotone_identity_skip() -> None:
    _nondecreasing(6, 6, "absolute")


def test_monotone_projection_skip() -> None:
    _nondecreasing(4, 6, "switch")
