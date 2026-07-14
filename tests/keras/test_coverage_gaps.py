# SPDX-License-Identifier: Apache-2.0
"""Cover keras kernel error paths and layer branches not hit elsewhere."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

from mononet.keras import (  # noqa: E402
    MonoInput,
    MonoResidual,
    _kernels,
)


def test_kernel_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _kernels.activation("bogus", ops.zeros(3))


def test_kernel_gate_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        _kernels.gate("bogus", ops.zeros(()))


def test_kernel_dense_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _kernels.monotonic_dense(
            ops.zeros((2, 3)), ops.ones((3, 4)), ops.zeros(4), "bogus", "relu"
        )


def test_residual_get_config_roundtrips() -> None:
    block = MonoResidual(4, mode="split", activation="relu")
    block(ops.zeros((2, 4)))  # build so config fields are populated
    cfg = block.get_config()
    assert cfg["units"] == 4
    assert cfg["mode"] == "split"
    assert cfg["activation"] == "relu"
    assert cfg["alpha_gate"] == "shifted_elu"
    assert cfg["beta_gate"] == "softplus"


def test_mono_input_accepts_scalar_direction() -> None:
    layer = MonoInput(-1)
    x = ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
    assert bool(ops.all(layer(x) == -x))
