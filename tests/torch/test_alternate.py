import numpy as np
import pytest
import torch
from torch import nn

from mononet.core.types import ActivationName
from mononet.torch import MonoLinear


def _stack(
    act: ActivationName = "relu", depth: int = 4, d: int = 4, h: int = 16
) -> nn.Sequential:
    torch.manual_seed(0)
    layers: list[MonoLinear] = []
    prev: MonoLinear | None = None
    prev_in = d
    for _ in range(depth):
        lay = MonoLinear(prev_in, h, mode="alternate", activation=act, prev=prev)
        layers.append(lay)
        prev, prev_in = lay, h
    layers.append(MonoLinear(prev_in, 1, mode="mixed", activation="identity"))
    return nn.Sequential(*layers)


def _alternate_layers(net: nn.Sequential) -> list[MonoLinear]:
    return [m for m in net if isinstance(m, MonoLinear) and m.mode == "alternate"]


def test_prev_alternates_phase_and_entry_is_convex() -> None:
    net = _stack()
    alt = _alternate_layers(net)
    assert [m._alt_convex for m in alt] == [True, False, True, False]


def test_entry_bias_zero_interior_bias_alternates_sign() -> None:
    net = _stack()
    alt = _alternate_layers(net)
    assert alt[0].bias is not None
    assert alt[1].bias is not None
    assert alt[2].bias is not None
    assert alt[0].bias.detach().abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert alt[1].bias.detach().mean().item() < 0.0  # concave interior
    assert alt[2].bias.detach().mean().item() > 0.0  # convex interior


def test_prev_not_retained() -> None:
    net = _stack()
    alt = _alternate_layers(net)
    assert all("prev" not in vars(m) for m in alt)
    assert not any("prev" in k for k in net.state_dict())


def test_alternate_is_monotone_nondecreasing() -> None:
    net = _stack()
    x = torch.zeros(1, 4)
    with torch.no_grad():
        base = net(x)
        for j in range(4):
            bumped = x.clone()
            bumped[0, j] += 1e-2
            assert (net(bumped) - base).item() >= -1e-5


def test_convex_fraction_rejected_for_alternate() -> None:
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoLinear(4, 8, mode="alternate", activation="relu", convex_fraction=0.3)


def test_prev_rejected_for_non_alternate() -> None:
    entry = MonoLinear(4, 8, mode="alternate", activation="relu")
    with pytest.raises(ValueError, match="prev"):
        MonoLinear(8, 8, mode="mixed", activation="relu", prev=entry)


def test_prev_must_be_alternate() -> None:
    mixed = MonoLinear(4, 8, mode="mixed", activation="relu")
    with pytest.raises(ValueError, match="alternate"):
        MonoLinear(8, 8, mode="alternate", activation="relu", prev=mixed)


def test_deep_alternate_trains_stably() -> None:
    # depth-8 plain alternate stack does not diverge (contrast: mixed diverges).
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.uniform(-1, 1, (2000, 4)), dtype=torch.float32)
    y = torch.tensor(
        (1 / (1 + np.exp(-3 * (x.numpy() - 0.1)))).sum(1, keepdims=True),
        dtype=torch.float32,
    )
    y = (y - y.mean()) / y.std()
    net = _stack(depth=8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lf = nn.MSELoss()
    for _ in range(300):
        opt.zero_grad()
        lf(net(x), y).backward()
        opt.step()
    with torch.no_grad():
        assert lf(net(x), y).item() < 0.9  # beats predict-the-mean (~1.0)


def test_mono_residual_rejects_alternate() -> None:
    from mononet.torch import MonoResidual

    with pytest.raises(ValueError, match="alternate"):
        MonoResidual(8, 8, mode="alternate", activation="relu")
