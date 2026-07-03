import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from benchmarks._common.init_diagnostics import (  # noqa: E402
    build_residual_stack,
    synthetic_monotone,
)


def _final_mse(
    sub_depth: int | None, *, depth: int = 32, epochs: int = 200, seed: int = 0
) -> float:
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack("absolute", depth, sub_depth)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    return loss_val


def test_deep_residual_trains_where_plain_fails() -> None:
    # depth-32 absolute: sub_depth=2 residual trains (~0.10 measured); plain diverges.
    residual = _final_mse(2)
    plain = _final_mse(None)
    assert residual < 0.3, f"residual d32 mse {residual}"
    assert plain > 1.0, f"plain d32 mse {plain}"
