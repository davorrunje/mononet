from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.examples._loader import load_example  # noqa: E402


def test_risk_net_forward_and_monotone() -> None:
    """RiskNet runs and is monotone in x_mono (dirs +1, +1, -1), free in x_free."""
    mod = load_example("risk_net_torch.py")
    torch.manual_seed(0)
    net = mod.RiskNet()
    x_mono = torch.randn(16, 3)
    x_free = torch.randn(16, 2)
    y = net(x_mono, x_free)
    assert tuple(y.shape) == (16, 1)
    with torch.no_grad():
        base = net(x_mono, x_free)
        for j, sign in ((0, 1), (1, 1), (2, -1)):
            bumped = x_mono.clone()
            bumped[:, j] += 0.5
            diff = (net(bumped, x_free) - base).squeeze(-1)
            assert bool(diff.abs().max() > 1e-4)  # genuinely responds to x_mono
            if sign > 0:
                assert bool((diff >= -1e-4).all())
            else:
                assert bool((diff <= 1e-4).all())
