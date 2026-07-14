import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402

from benchmarks._common.init_diagnostics import (  # noqa: E402
    build_residual_stack,
    synthetic_monotone,
)
from mononet.torch import MonoResidual, _kernels  # noqa: E402


def _final_mse(
    sub_depth: int | None, *, depth: int = 32, epochs: int = 200, seed: int = 0
) -> float:
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack("mixed", depth, sub_depth)
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


def _train_default_deep(
    depth: int = 16, epochs: int = 200, seed: int = 0
) -> tuple[float, float, int, int]:
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack("mixed", depth, 2)  # sub_depth=2, A+B defaults
    blocks = [m for m in net.modules() if isinstance(m, MonoResidual)]
    w0 = [float(_last_weight_abs_sum(b)) for b in blocks]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    g_beta = [float(_kernels.gate(b.beta_gate, b.beta).detach()) for b in blocks]
    moved = sum(
        1
        for b, w in zip(blocks, w0, strict=True)
        if abs(float(_last_weight_abs_sum(b)) - w) > 1e-9
    )
    return loss_val, max(g_beta), moved, len(blocks)


def _last_weight_abs_sum(block: MonoResidual) -> float:
    from mononet.torch import MonoLinear

    f = block.F
    last = f if isinstance(f, MonoLinear) else f[-1]  # type: ignore[index]
    return float(last.weight.detach().abs().sum())


def test_deep_default_uses_depth() -> None:
    loss, max_g_beta, moved, n = _train_default_deep()
    # Trap-1 guard: the softplus gate opens (impossible under the old scaled_elu
    # dead zone, where g_beta stays ~1e-3). Verified red on main.
    assert max_g_beta > 0.1, f"gate did not open: max g_beta {max_g_beta}"
    # Trap-2 guard: F's last-layer weights actually train (exact-zero init would
    # freeze them at 0 under |W|, making F a constant).
    assert moved == n, f"only {moved}/{n} blocks' F weights moved"
    # Stays bounded (non-divergent). NOTE: this shallow separable target does not
    # need depth, so engaged depth settles ~0.8 here rather than a low floor —
    # the depth-null result; beneficial depth on adequate data is shown by
    # benchmarks/monoresidual_gate_ablation.py. This only guards engagement+stability.
    assert loss < 2.0, f"deep default not bounded: {loss}"
