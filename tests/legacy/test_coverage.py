# SPDX-License-Identifier: Apache-2.0
"""Coverage-closing tests for validation/branch paths in the legacy port.

Targets the builder validation errors, the convexity-params checks, the
``replace_kernel_using_monotonicity_indicator`` context manager, and the
dropout/final-activation/explicit-input-units branches that the main legacy
test files don't otherwise exercise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy.mono_dense_layer import (  # noqa: E402
    _check_convexity_params,
    _create_mono_block,
    _prepare_mono_input_n_param,
    apply_activations,
    get_activation_functions,
    replace_kernel_using_monotonicity_indicator,
)


class _FakeLayer:
    """Minimal stand-in for a Keras layer: a mutable ``kernel`` attribute."""

    def __init__(self, kernel: object) -> None:
        self.kernel = kernel


# --- apply_activations -------------------------------------------------


def test_apply_activations_rejects_short_weight_tuple() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.zeros((1, 4))
    with pytest.raises(ValueError, match="activation_weights"):
        apply_activations(
            x,
            units=4,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            activation_weights=(1.0, 1.0),  # type: ignore[arg-type]
        )


# --- replace_kernel_using_monotonicity_indicator ------------------------


def test_replace_kernel_context_manager_constrains_and_restores() -> None:
    original_kernel = ops.convert_to_tensor(
        np.array([[-1.0, 2.0], [3.0, -4.0]], dtype="float32")
    )
    layer = _FakeLayer(original_kernel)
    indicator = ops.convert_to_tensor(np.array([[1], [-1]], dtype="float32"))

    with replace_kernel_using_monotonicity_indicator(layer, indicator):
        constrained = np.asarray(layer.kernel)
        assert np.allclose(constrained, [[1.0, 2.0], [-3.0, -4.0]])

    restored = np.asarray(layer.kernel)
    assert np.allclose(restored, [[-1.0, 2.0], [3.0, -4.0]])


# --- _create_mono_block ---------------------------------------------------


def test_create_mono_block_empty_units_is_identity() -> None:
    x = ops.convert_to_tensor(np.array([[1.0, 2.0]], dtype="float32"))
    block = _create_mono_block(units=[], activation="relu")
    assert block(x) is x


def test_create_mono_block_dropout_inserts_dropout_layer() -> None:
    legacy._WARNED = True
    inp = keras.Input(shape=(3,))
    block = _create_mono_block(units=[4, 2], activation="relu", dropout=0.5)
    out = block(inp)
    model = keras.Model(inp, out)
    assert any(isinstance(layer, keras.layers.Dropout) for layer in model.layers)


# --- _prepare_mono_input_n_param -----------------------------------------


def test_prepare_mono_input_n_param_list_length_mismatch() -> None:
    inputs = [ops.zeros((1, 1)), ops.zeros((1, 1))]
    with pytest.raises(ValueError, match=r"2 != 1"):
        _prepare_mono_input_n_param(inputs, [1])


def test_prepare_mono_input_n_param_list_incompatible_type() -> None:
    inputs = [ops.zeros((1, 1)), ops.zeros((1, 1))]
    with pytest.raises(ValueError, match="Incompatible types"):
        _prepare_mono_input_n_param(inputs, "bad")


def test_prepare_mono_input_n_param_dict_key_mismatch() -> None:
    inputs = {"a": ops.zeros((1, 1)), "b": ops.zeros((1, 1))}
    with pytest.raises(ValueError, match=r"!="):
        _prepare_mono_input_n_param(inputs, {"a": 1, "c": 1})


def test_prepare_mono_input_n_param_dict_incompatible_type() -> None:
    inputs = {"a": ops.zeros((1, 1))}
    with pytest.raises(ValueError, match="Incompatible types"):
        _prepare_mono_input_n_param(inputs, "bad")


def test_prepare_mono_input_n_param_bare_tensor_non_int_param() -> None:
    x = ops.zeros((1, 1))
    with pytest.raises(ValueError, match="Incompatible types"):
        _prepare_mono_input_n_param(x, [1])


# --- _check_convexity_params -----------------------------------------------


def test_check_convexity_params_raises_both_convex_and_concave() -> None:
    with pytest.raises(ValueError, match="both convex and concave"):
        _check_convexity_params([1, 1], [True, False], [True, False], ["a", "b"])


def test_check_convexity_params_warns_when_both_present_on_different_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    has_convex, has_concave = _check_convexity_params(
        [1, -1], [True, False], [False, True], ["a", "b"]
    )
    assert has_convex is True
    assert has_concave is True
    captured = capsys.readouterr()
    assert "both convex and concave" in captured.out


# --- create_type_1 / create_type_2 ---------------------------------------


def test_create_type_1_final_activation_sigmoid_squashes_output() -> None:
    legacy._WARNED = True
    inp = keras.Input(shape=(3,))
    out = legacy.create_type_1(
        inp,
        units=4,
        final_units=1,
        activation="relu",
        n_layers=2,
        final_activation="sigmoid",
    )
    model = keras.Model(inp, out)
    y = np.asarray(model(np.array([[10.0, 10.0, 10.0]], dtype="float32")))
    assert 0.0 <= y[0, 0] <= 1.0


def test_create_type_2_explicit_input_units_dropout_concave_final_activation() -> None:
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(2)]
    out = legacy.create_type_2(
        inputs,
        input_units=3,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=2,
        monotonicity_indicator=[1, -1],
        is_concave=[True, True],
        dropout=0.5,
        final_activation="sigmoid",
    )
    model = keras.Model(inputs, out)
    xs = [np.zeros((2, 1), dtype="float32") for _ in range(2)]
    y = np.asarray(model(xs))
    assert y.shape == (2, 1)
    assert np.all((y >= 0.0) & (y <= 1.0))
