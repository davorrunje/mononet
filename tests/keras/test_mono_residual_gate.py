"""Near-zero init of the default F, and gate defaults, for MonoResidual (keras)."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

from mononet.keras import MonoDense, MonoResidual  # noqa: E402


def _last_dense(block: MonoResidual) -> MonoDense:
    f = block.F
    if isinstance(f, MonoDense):
        return f
    last = f.layers[-1]  # keras.Sequential
    assert isinstance(last, MonoDense)
    return last


def test_default_F_last_layer_is_near_zero_but_nonzero() -> None:  # noqa: N802
    keras.utils.set_random_seed(0)
    block = MonoResidual(32, mode="mixed", activation="elu")
    block(np.zeros((1, 32), dtype="float32"))  # build
    last = _last_dense(block)
    wnorm = float(ops.convert_to_numpy(ops.sum(ops.abs(last.w))))
    # small but NOT exactly zero (exact zero would freeze under |W|)
    assert wnorm > 0.0
    assert wnorm < 1.0  # heavily attenuated vs a normal init (~tens)
    # bias zeroed
    assert last.b is not None
    assert float(ops.convert_to_numpy(ops.sum(ops.abs(last.b)))) == 0.0


def test_default_block_is_near_identity_at_init() -> None:
    keras.utils.set_random_seed(0)
    block = MonoResidual(32, mode="mixed", activation="elu")
    x = ops.convert_to_tensor(
        np.random.default_rng(0).standard_normal((8, 32)).astype("float32")
    )
    fx = block.F(x)  # builds F too
    fx_rms = float(ops.convert_to_numpy(ops.sqrt(ops.mean(ops.square(fx)))))
    assert fx_rms < 0.2  # F(x) ~= 0 at init => block ~= g_alpha * skip


def test_custom_F_is_not_near_zeroed() -> None:  # noqa: N802
    keras.utils.set_random_seed(0)
    custom = MonoDense(32, mode="mixed", activation="elu")
    custom(np.zeros((1, 32), dtype="float32"))  # build
    before = float(ops.convert_to_numpy(ops.sum(ops.abs(custom.w))))
    block = MonoResidual(32, F=custom)
    block(np.zeros((1, 32), dtype="float32"))
    after = float(ops.convert_to_numpy(ops.sum(ops.abs(block.F.w))))  # type: ignore[union-attr]
    assert after == before  # untouched


def test_near_zero_scale_is_user_tunable() -> None:
    keras.utils.set_random_seed(0)
    small_block = MonoResidual(32, mode="mixed", activation="elu")
    small_block(np.zeros((1, 32), dtype="float32"))
    small = _last_dense(small_block)

    keras.utils.set_random_seed(0)
    big_block = MonoResidual(
        32, mode="mixed", activation="elu", near_zero_scale=2e-3
    )
    big_block(np.zeros((1, 32), dtype="float32"))
    big = _last_dense(big_block)

    # same seed => 2e-3 gives ~2x the weight magnitude of the 1e-3 default
    small_norm = float(ops.convert_to_numpy(ops.sum(ops.abs(small.w))))
    big_norm = float(ops.convert_to_numpy(ops.sum(ops.abs(big.w))))
    ratio = big_norm / small_norm
    assert ratio == pytest.approx(2.0, rel=1e-5)

    # 0.0 reproduces exact-zero
    keras.utils.set_random_seed(0)
    zero_block = MonoResidual(
        32, mode="mixed", activation="elu", near_zero_scale=0.0
    )
    zero_block(np.zeros((1, 32), dtype="float32"))
    zero = _last_dense(zero_block)
    assert float(ops.convert_to_numpy(ops.sum(ops.abs(zero.w)))) == 0.0


def test_near_zero_scale_with_bias_false() -> None:
    # covers the no-bias branch of near-zero init: weight scaled, no bias to zero
    layer = MonoDense(
        4, mode="mixed", activation="elu", bias=False, near_zero_scale=1e-3
    )
    layer(np.zeros((2, 4), dtype="float32"))  # triggers build with no bias
    assert layer.b is None
    assert float(ops.convert_to_numpy(ops.sum(ops.abs(layer.w)))) > 0.0
