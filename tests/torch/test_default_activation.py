from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear  # noqa: E402


def test_default_activation_is_affine() -> None:
    # identity default => affine map => midpoint-preserving.
    layer = MonoLinear(4, 8, mode="split")
    x1 = torch.randn(5, 4)
    x2 = torch.randn(5, 4)
    mid = layer((x1 + x2) / 2)
    avg = (layer(x1) + layer(x2)) / 2
    torch.testing.assert_close(mid, avg, rtol=1e-5, atol=1e-5)
