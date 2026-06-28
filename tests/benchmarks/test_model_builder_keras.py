import numpy as np
import pytest

pytest.importorskip("keras")

import keras

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.model_builder import build_model


def _bundle(n: int = 64, d: int = 7) -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = (X[:, 4] * -1.0 + rng.normal(scale=0.1, size=n)).astype(np.float64)
    return DatasetBundle(
        name="syn",
        task="regression",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        mono_increasing=(),
        mono_decreasing=(4,),
        feature_names=tuple(f"f{i}" for i in range(d)),
        metadata={},
    )


def _cfg(mode: str, residual: bool) -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset="syn",
        backend="keras",
        mode=mode,
        residual=residual,
        depth=2,
        width=8,
        activation="elu",
        convex_fraction=0.5,
        embed_hidden=(8,),
        dropout=0.0,
        optimizer=OptimizerSpec("adam", 1e-3, 0.0),
        lr_decay=None,
        batch_size=16,
        epochs=1,
        early_stopping=None,
        seeds=(0,),
        metrics=("mse",),
    )


@pytest.mark.parametrize("mode", ["switch", "absolute"])
@pytest.mark.parametrize("residual", [False, True])
def test_builds_and_output_shape(mode: str, residual: bool) -> None:
    b = _bundle()
    model = build_model(_cfg(mode, residual), b)
    x = keras.ops.convert_to_tensor(b.X_train)
    out = model(x)
    assert tuple(out.shape) == (b.X_train.shape[0], 1)


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_monotone_in_decreasing_feature(mode: str) -> None:
    # Output must be non-increasing in column 4 (declared decreasing).
    b = _bundle()
    model = build_model(_cfg(mode, residual=False), b)
    x = b.X_train.copy()
    x_hi = x.copy()
    x_hi[:, 4] += 1.0
    out = np.array(model(keras.ops.convert_to_tensor(x)))
    out_hi = np.array(model(keras.ops.convert_to_tensor(x_hi)))
    assert np.all(out_hi <= out + 1e-5)
