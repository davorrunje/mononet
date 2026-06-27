"""Contract test for the mononet.keras public API surface."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402


def test_exports() -> None:
    import mononet.keras as kmod

    assert set(kmod.__all__) == {"MonoDense", "MonoResidual", "MonoInput"}


def test_mono_dense_runs_and_serializes() -> None:
    import mononet.keras as kmod

    layer = kmod.MonoDense(5, mode="absolute", convex_fraction=0.25)
    y = layer(ops.zeros((2, 3)))
    assert tuple(y.shape) == (2, 5)
    cfg = layer.get_config()
    clone = kmod.MonoDense.from_config(cfg)
    assert clone.mode == "absolute"
    assert clone.convex_fraction == 0.25


def test_mono_residual_warm_start_near_identity() -> None:
    import mononet.keras as kmod

    block = kmod.MonoResidual(4, mode="switch")
    x = ops.convert_to_tensor(np.random.default_rng(0).normal(size=(3, 4)))
    assert bool(ops.all(ops.abs(block(x) - x) < 5e-3))


def test_mono_input_flips_signs() -> None:
    import mononet.keras as kmod
    from mononet.core.types import MonotonicityMask

    layer = kmod.MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    x = ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
    assert bool(
        ops.all(layer(x) == ops.convert_to_tensor(np.array([[1.0, -2.0, 3.0]])))
    )


def test_mono_input_mask_serializes() -> None:
    import mononet.keras as kmod
    from mononet.core.types import MonotonicityMask

    layer = kmod.MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    cfg = layer.get_config()
    clone = kmod.MonoInput.from_config(cfg)
    x = ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
    expected = ops.convert_to_tensor(np.array([[1.0, -2.0, 3.0]]))
    assert bool(ops.all(clone(x) == expected))
