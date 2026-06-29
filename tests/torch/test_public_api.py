# SPDX-License-Identifier: Apache-2.0
"""Contract test for the mononet.torch public API surface."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_exports() -> None:
    import mononet.torch as t

    assert set(t.__all__) == {"MonoLinear", "MonoResidual", "MonoInput"}


def test_mono_linear_is_module_and_runs() -> None:
    import mononet.torch as t

    layer = t.MonoLinear(3, 5, mode="switch")
    assert isinstance(layer, torch.nn.Module)
    y = layer(torch.zeros(2, 3))
    assert y.shape == (2, 5)


def test_mono_residual_warm_start_near_identity() -> None:
    import mononet.torch as t

    torch.manual_seed(0)  # deterministic init + input (jax/keras tests already seed)
    block = t.MonoResidual(4, 4, mode="switch")
    x = torch.randn(3, 4)
    y = block(x)
    assert torch.allclose(y, x, atol=5e-3)


def test_mono_input_flips_signs() -> None:
    import mononet.torch as t
    from mononet.core.types import MonotonicityMask

    layer = t.MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(layer(x), torch.tensor([[1.0, -2.0, 3.0]]))
