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


def test_deep_residual_stays_bounded_where_plain_diverges() -> None:
    # Residual skips keep a depth-32 stack BOUNDED where a plain stack diverges.
    # With the A+B default (softplus gate + near-zero-init F) the residual branch
    # now genuinely engages (gate open, F trains) instead of sitting idle, so on
    # this shallow *separable* target (y = Σ softplus(aᵢ·xᵢ), which needs no
    # depth) at this deliberately small/fast budget it settles ~0.8 rather than
    # the old F-off skip-only ~0.10 — engaged depth is neutral-to-harmful here
    # (the depth-null result). It stays BOUNDED, unlike the plain stack (~1e23).
    # That engaged depth *does* train to low MSE given an adequate budget is
    # shown by benchmarks/monoresidual_gate_ablation.py (0.068) and confirmed at
    # n=4000/lr=1e-3 (0.138); here we only guard trainability-vs-divergence.
    residual = _final_mse(2)
    plain = _final_mse(None)
    assert residual < 2.0, f"residual d32 not bounded: {residual}"
    assert plain > 100.0, f"plain d32 did not diverge: {plain}"
