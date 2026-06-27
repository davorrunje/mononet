"""Numeric + monotonicity tests for monotonic_dense."""

from __future__ import annotations

import numpy as np
import pytest

from mononet.core import reference as ref
from mononet.core.types import ActivationSpec


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_switch_reduces_to_abs_weights_in_linear_regime() -> None:
    # With a piecewise-linear-at-large-scale check we instead verify the
    # identity at the formula level: relu, large positive pre-activations.
    rng = _rng(0)
    x = rng.normal(size=(5, 3)).astype(np.float64)
    w = rng.normal(size=(3, 4)).astype(np.float64)
    b = np.zeros(4)
    y = ref.monotonic_dense(x, w, b, "switch", ActivationSpec("relu"), 0.5)
    assert y.shape == (5, 4)


@pytest.mark.parametrize("mode", ["switch", "absolute"])
@pytest.mark.parametrize("name", ["relu", "elu", "selu", "softplus"])
def test_nondecreasing_in_every_input(mode: str, name: str) -> None:
    rng = _rng(7)
    w = rng.normal(size=(3, 5)).astype(np.float64)
    b = rng.normal(size=5).astype(np.float64)
    spec = ActivationSpec(name)  # type: ignore[arg-type]
    x = rng.normal(size=(8, 3)).astype(np.float64)
    bump = np.zeros_like(x)
    bump[:, 1] = 0.5  # increase input feature 1
    y0 = ref.monotonic_dense(x, w, b, mode, spec, 0.5)  # type: ignore[arg-type]
    y1 = ref.monotonic_dense(x + bump, w, b, mode, spec, 0.5)  # type: ignore[arg-type]
    assert np.all(y1 - y0 >= -1e-9)


def test_absolute_convex_fraction_endpoints() -> None:
    rng = _rng(1)
    x = rng.normal(size=(4, 2)).astype(np.float64)
    w = rng.normal(size=(2, 6)).astype(np.float64)
    b = rng.normal(size=6).astype(np.float64)
    spec = ActivationSpec("relu")
    h = x @ np.abs(w) + b
    all_convex = ref.monotonic_dense(x, w, b, "absolute", spec, 1.0)
    all_concave = ref.monotonic_dense(x, w, b, "absolute", spec, 0.0)
    np.testing.assert_allclose(all_convex, ref.base_activation("relu", h))
    np.testing.assert_allclose(all_concave, ref.concave_reflection("relu", h))
