# SPDX-License-Identifier: Apache-2.0
"""Unit tests for legacy helper functions."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

from mononet.legacy.mono_dense_layer import (  # noqa: E402
    apply_activations,
    apply_monotonicity_indicator_to_kernel,
    get_activation_functions,
    get_monotonicity_indicator,
)


def test_get_activation_functions_relu_convex_concave() -> None:
    convex, concave, _ = get_activation_functions("relu")
    x = ops.convert_to_tensor(np.array([-2.0, -1.0, 1.0, 2.0], dtype="float32"))
    # convex = relu; concave(x) = -relu(-x)
    assert np.allclose(np.asarray(convex(x)), [0.0, 0.0, 1.0, 2.0])
    assert np.allclose(np.asarray(concave(x)), [-2.0, -1.0, 0.0, 0.0])


def test_saturated_activation_is_continuous_at_zero() -> None:
    _, _, saturated = get_activation_functions("elu")
    x = ops.convert_to_tensor(np.array([-1e-6, 0.0, 1e-6], dtype="float32"))
    y = np.asarray(saturated(x))
    assert abs(y[0] - y[2]) < 1e-3  # continuous across the x<=0 boundary


def test_apply_activations_convex_uses_all_convex_split() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.convert_to_tensor(np.array([[-1.0, -2.0, 3.0]], dtype="float32"))
    y = np.asarray(
        apply_activations(
            x,
            units=3,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            is_convex=True,
        )
    )
    assert np.allclose(y, [[0.0, 0.0, 3.0]])  # all-convex == relu


def test_apply_activations_weighted_split_sizes() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.convert_to_tensor(np.arange(10, dtype="float32").reshape(1, 10))
    # weights (7,7,2)/16 * 10 -> round(4.375)=4, round(4.375)=4, remainder=2
    y = np.asarray(
        apply_activations(
            x,
            units=10,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            activation_weights=(7.0, 7.0, 2.0),
        )
    )
    assert y.shape == (1, 10)


def test_apply_activations_rejects_bad_weights() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.zeros((1, 4))
    with pytest.raises(ValueError, match="activation_weights"):
        apply_activations(
            x,
            units=4,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            activation_weights=(1.0, -1.0, 1.0),
        )


def test_get_monotonicity_indicator_reshapes_to_column() -> None:
    ind = get_monotonicity_indicator([1, -1, 0], input_shape=(None, 3), units=4)
    assert ind.shape == (3, 1)


def test_get_monotonicity_indicator_rejects_out_of_domain() -> None:
    with pytest.raises(ValueError, match="must be one of -1, 0, 1"):
        get_monotonicity_indicator([2], input_shape=(None, 1), units=1)


def test_get_monotonicity_indicator_rejects_rank_gt_2() -> None:
    with pytest.raises(ValueError, match="rank greater than 2"):
        get_monotonicity_indicator(np.ones((2, 2, 2)), input_shape=(None, 2), units=2)


def test_apply_indicator_to_kernel_signs() -> None:
    kernel = ops.convert_to_tensor(
        np.array([[-1.0, 2.0], [3.0, -4.0]], dtype="float32")
    )
    indicator = ops.convert_to_tensor(np.array([[1], [-1]], dtype="float32"))
    out = np.asarray(apply_monotonicity_indicator_to_kernel(kernel, indicator))
    # row 0 -> |.| (increasing); row 1 -> -|.| (decreasing)
    assert np.allclose(out, [[1.0, 2.0], [-3.0, -4.0]])


def test_apply_indicator_zero_leaves_kernel_unchanged() -> None:
    kernel = ops.convert_to_tensor(np.array([[-1.0, 2.0]], dtype="float32"))
    indicator = ops.convert_to_tensor(np.array([[0]], dtype="float32"))
    out = np.asarray(apply_monotonicity_indicator_to_kernel(kernel, indicator))
    assert np.allclose(out, [[-1.0, 2.0]])
