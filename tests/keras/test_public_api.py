"""Contract test for the mononet.keras public API surface."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")


def test_mono_dense_exists_and_is_keras_layer() -> None:
    from mononet.keras import MonoDense

    assert issubclass(MonoDense, keras.layers.Layer)


def test_mono_mlp_exists_and_is_keras_model() -> None:
    from mononet.keras import MonoMLP

    assert issubclass(MonoMLP, keras.Model)


def test_instantiating_mono_dense_raises_not_implemented() -> None:
    import numpy as np

    from mononet.core.types import ActivationSpec, MonotonicityMask
    from mononet.keras import MonoDense

    with pytest.raises(NotImplementedError):
        MonoDense(
            units=4,
            monotonicity=MonotonicityMask(np.zeros(8, dtype=np.int8)),
            activation=ActivationSpec(name="relu"),
        )


def test_no_unexpected_top_level_exports() -> None:
    import mononet.keras as k

    expected = {"MonoDense", "MonoMLP"}
    assert set(k.__all__) == expected
