# SPDX-License-Identifier: Apache-2.0
"""Cover torch kernel error paths and layer branches not hit elsewhere."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mononet.core.types import InitSpec  # noqa: E402
from mononet.torch import (  # noqa: E402
    MonoInput,
    MonoLinear,
    MonoResidual,
    _kernels,
)


def test_kernel_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _kernels.activation("bogus", torch.zeros(3))


def test_kernel_gate_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        _kernels.gate("bogus", torch.zeros(()))


def test_kernel_dense_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _kernels.monotonic_dense(
            torch.zeros(2, 3), torch.ones(3, 4), torch.zeros(4), "bogus", "relu"
        )


def test_init_weight_accepts_initspec_with_seed() -> None:
    # InitSpec instance (not None/str) hits the `spec = init` branch;
    # a non-None seed hits the `torch.manual_seed(spec.seed)` branch.
    layer = MonoLinear(
        3, 5, activation="relu", init=InitSpec(scheme="he_normal", seed=0)
    )
    assert layer(torch.zeros(2, 3)).shape == (2, 5)


def test_residual_accepts_callable_f_factory() -> None:
    # A plain callable (not an nn.Module) hits the `self.F = F(units)` branch.
    block = MonoResidual(4, 4, F=lambda u: MonoLinear(u, u, activation="relu"))
    assert block(torch.zeros(2, 4)).shape == (2, 4)


def test_mono_input_accepts_scalar_direction() -> None:
    # int direction (not a MonotonicityMask) hits the scalar branch.
    layer = MonoInput(-1)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(layer(x), -x)
