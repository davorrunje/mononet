import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from benchmarks._common.init_diagnostics import synthetic_monotone  # noqa: E402
from mononet.torch import MonoLinear  # noqa: E402


def _train_absolute(
    init: object, depth: int, *, epochs: int = 150, seed: int = 0
) -> float:
    """Train a depth-`depth` absolute MonoLinear stack; return final train MSE."""
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    width = 32
    layers: list[nn.Module] = [
        MonoLinear(8, width, mode="mixed", activation="elu", init=init)  # type: ignore[arg-type]
    ]
    layers += [
        MonoLinear(width, width, mode="mixed", activation="elu", init=init)  # type: ignore[arg-type]
        for _ in range(depth - 1)
    ]
    layers.append(
        MonoLinear(width, 1, mode="mixed", activation="elu", init=init)  # type: ignore[arg-type]
    )
    net = nn.Sequential(*[layer.double() for layer in layers])
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    return loss_val


def test_new_init_learns_where_he_normal_does_not() -> None:
    # depth-2 absolute: y is unit-variance, so <0.5 = learning, ~1.0+ = not learning.
    new = _train_absolute(None, depth=2)
    old = _train_absolute("he_normal", depth=2)
    assert new < 0.5 < old, f"new={new:.3f} old={old:.3f}"


def test_new_init_beats_he_normal_at_depth_4() -> None:
    # deeper (4): new init need not fully converge, but must clearly beat the old default.
    new = _train_absolute(None, depth=4)
    old = _train_absolute("he_normal", depth=4)
    assert new < old - 0.5, f"new={new:.3f} old={old:.3f}"
