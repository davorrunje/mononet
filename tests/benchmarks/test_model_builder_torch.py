from typing import Literal

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle  # noqa: E402
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec  # noqa: E402
from benchmarks._common.model_builder import build_model  # noqa: E402


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


def _cfg(mode: Literal["switch", "absolute"], residual: bool) -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset="syn",
        backend="torch",
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
def test_builds_and_output_shape(
    mode: Literal["switch", "absolute"], residual: bool
) -> None:
    b = _bundle()
    model = build_model(_cfg(mode, residual), b)
    x = torch.tensor(b.X_train, dtype=torch.float32)
    out = model(x)
    assert out.shape == (b.X_train.shape[0], 1)


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_monotone_in_decreasing_feature(mode: Literal["switch", "absolute"]) -> None:
    # Output must be non-increasing in column 4 (declared decreasing).
    b = _bundle()
    model = build_model(_cfg(mode, residual=False), b).eval()
    x = torch.tensor(b.X_train, dtype=torch.float32)
    x_hi = x.clone()
    x_hi[:, 4] += 1.0
    with torch.no_grad():
        assert torch.all(model(x_hi) <= model(x) + 1e-5)


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_head_is_linear_not_relu(mode: Literal["switch", "absolute"]) -> None:
    # Regression guard: the read-out head must be a *linear* monotone map
    # (identity activation). A nonlinear head (the MonoLinear default is ReLU)
    # forces the pre-sigmoid >= 0 in absolute mode, collapsing binary
    # classification to the base rate. See model_builder head construction.
    model = build_model(_cfg(mode, residual=False), _bundle())
    assert model.head.activation_name == "identity"


def _binary_bundle(n: int = 400, d: int = 6) -> DatasetBundle:
    # A learnable classification task: label depends on a monotone feature (0,
    # increasing) AND a *non-monotone* free feature (a centred quadratic).
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, d)).astype(np.float64)
    logit = 1.5 * X[:, 0] - 2.0 * (X[:, 3] ** 2 - 1.0)
    y = (logit + 0.1 * rng.normal(size=n) > 0).astype(np.float64)
    return DatasetBundle(
        name="synbin",
        task="binary_classification",
        X_train=X,
        y_train=y,
        X_test=X,
        y_test=y,
        mono_increasing=(0,),
        mono_decreasing=(),
        feature_names=tuple(f"f{i}" for i in range(d)),
        metadata={},
    )


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_binary_classification_beats_base_rate(
    mode: Literal["switch", "absolute"],
) -> None:
    # End-to-end guard against the ReLU-head base-rate collapse: a short train
    # run must clear the majority-class baseline in both modes.
    import dataclasses

    from benchmarks._common.runner import run

    b = _binary_bundle()
    base = max(float(b.y_test.mean()), 1.0 - float(b.y_test.mean()))
    cfg = dataclasses.replace(
        _cfg(mode, residual=False), epochs=40, metrics=("accuracy",)
    )
    rows = run(cfg, b)
    assert rows[0].scores["accuracy"] > base + 0.03
