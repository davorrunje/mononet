"""Tests for XGBoost baseline."""

import numpy as np
import pytest

# xgboost can fail to import for reasons other than absence -- notably on macOS
# its wheel needs libomp.dylib, which raises XGBoostError (not ImportError) when
# missing. importorskip only catches ImportError, so skip on any import failure.
try:
    import xgboost  # noqa: F401
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"xgboost unavailable: {exc}", allow_module_level=True)

from benchmarks._common.bundle import DatasetBundle
from benchmarks.baselines.xgboost import run_xgboost


def test_xgboost_regression_finite_mse() -> None:
    """Test XGBoost regression returns finite MSE."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = X[:, 0] + 0.1 * rng.normal(size=200)
    b = DatasetBundle(
        "syn",
        "regression",
        X,
        y,
        X,
        y,
        (0,),
        (),
        tuple(f"f{i}" for i in range(5)),
        {},
    )
    scores = run_xgboost(b, seed=0)
    assert np.isfinite(scores["mse"])
    assert np.isfinite(scores["rmse"])


def test_xgboost_binary_classification_finite_accuracy() -> None:
    """Test XGBoost binary classification returns finite accuracy in [0, 1]."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    # Create binary target: 0 and 1
    y = (X[:, 0] > 0).astype(int)
    b = DatasetBundle(
        "syn",
        "binary_classification",
        X,
        y,
        X,
        y,
        (),
        (),
        tuple(f"f{i}" for i in range(5)),
        {},
    )
    scores = run_xgboost(b, seed=0)
    assert np.isfinite(scores["accuracy"])
    assert 0 <= scores["accuracy"] <= 1
