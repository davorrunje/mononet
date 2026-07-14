from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.init_diagnostics import (
    grad_flow,
    synthetic_monotone,
    trainability,
)

if TYPE_CHECKING:
    from mononet.core.config import Mode


def test_synthetic_monotone_shapes_and_standardized() -> None:
    X, y = synthetic_monotone(256, 5, seed=0)
    assert X.shape == (256, 5)
    assert y.shape == (256,)
    assert abs(float(X.mean())) < 0.1
    assert abs(float(X.std()) - 1.0) < 0.1


@pytest.mark.parametrize("mode", ["split", "mixed"])
def test_grad_flow_finite(mode: Mode) -> None:
    out = grad_flow(mode, depth=4, activation="elu", width=16, seed=0)
    assert np.isfinite(out["input_grad_norm"])
    layer_norms = out["layer_grad_norms"]
    assert isinstance(layer_norms, list)
    assert len(layer_norms) == 4
    assert all(np.isfinite(g) for g in layer_norms)


@pytest.mark.parametrize("mode", ["split", "mixed"])
def test_trainability_finite(mode: Mode) -> None:
    out = trainability(mode, depth=2, activation="elu", epochs=3, seed=0)
    assert np.isfinite(out["final_train_loss"])
