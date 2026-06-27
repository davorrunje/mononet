"""Tests for the reference activations, reflection, and gates."""

from __future__ import annotations

import numpy as np
import pytest

from mononet.core import reference as ref


def test_relu_and_softplus() -> None:
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(ref.base_activation("relu", x), [0.0, 0.0, 3.0])
    np.testing.assert_allclose(
        ref.base_activation("softplus", x), np.log1p(np.exp(x)), atol=1e-6
    )


def test_concave_reflection_is_minus_act_of_minus_x() -> None:
    x = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(
        ref.concave_reflection("relu", x), -ref.base_activation("relu", -x)
    )


@pytest.mark.parametrize("name", ["relu", "elu", "selu", "softplus"])
def test_activations_are_nondecreasing(name: str) -> None:
    # Note: gelu is excluded — the tanh approximation (and true GELU) has a
    # local minimum near x ≈ -0.75 and is not globally non-decreasing.
    x = np.linspace(-5, 5, 200)
    y = ref.base_activation(name, x)  # type: ignore[arg-type]
    assert np.all(np.diff(y) >= -1e-7)


def test_gate_values_and_derivatives_at_zero() -> None:
    zero = np.array(0.0)
    assert ref.apply_gate("shifted_elu", zero) == pytest.approx(1.0)
    assert ref.apply_gate("scaled_elu", zero) == pytest.approx(1e-3)
    # finite-difference derivative at 0 is ~1 for both
    h = 1e-6
    for token in ("shifted_elu", "scaled_elu"):
        d = (
            ref.apply_gate(token, np.array(h)) - ref.apply_gate(token, np.array(-h))
        ) / (2 * h)
        assert d == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("token", ["shifted_elu", "scaled_elu"])
def test_gates_are_strictly_positive(token: str) -> None:
    x = np.linspace(-10, 10, 100)
    assert np.all(ref.apply_gate(token, x) > 0.0)
