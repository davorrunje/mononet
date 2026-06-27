# SPDX-License-Identifier: Apache-2.0
"""Property test: PyTorch layers are non-decreasing in every input."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from mononet.torch import MonoLinear, MonoResidual  # noqa: E402


@settings(deadline=None, max_examples=50)
@given(
    mode=st.sampled_from(["switch", "absolute"]),
    seed=st.integers(0, 10_000),
)
def test_mono_linear_nondecreasing(mode: str, seed: int) -> None:
    torch.manual_seed(seed)
    layer = MonoLinear(3, 4, mode=mode)
    x = torch.randn(6, 3)
    bump = torch.zeros_like(x)
    bump[:, 1] = torch.rand(6).abs() + 0.1
    with torch.no_grad():
        delta = layer(x + bump) - layer(x)
    assert torch.all(delta >= -1e-5)


@settings(deadline=None, max_examples=30)
@given(seed=st.integers(0, 10_000))
def test_mono_residual_nondecreasing(seed: int) -> None:
    torch.manual_seed(seed)
    block = MonoResidual(3, 3, mode="switch")
    # perturb the gate raws away from the warm start
    with torch.no_grad():
        block.beta.add_(torch.rand(()) * 2)
        block.alpha.add_(torch.rand(()) * 2)
    x = torch.randn(6, 3)
    bump = torch.zeros_like(x)
    bump[:, 0] = torch.rand(6).abs() + 0.1
    with torch.no_grad():
        delta = block(x + bump) - block(x)
    assert torch.all(delta >= -1e-5)
