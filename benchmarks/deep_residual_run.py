"""Skip-K trainability + conditioning sweep for deep monotone residual stacks.

Writes ``benchmarks/results/deep-residual/trainability.json`` (committed; read by
``docs/concepts/monotonic-residual.md``). Repo-only; never shipped in the wheel.

Run: ``uv run --extra torch --group bench python -m benchmarks.deep_residual_run``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks._common.init_diagnostics import build_residual_stack, synthetic_monotone

_MODES = ("absolute", "switch")
_DEPTHS = (4, 8, 16, 32)
_KS: tuple[int | None, ...] = (None, 1, 2, 4, 8)
_CAP = 1.0e6


def _run(
    mode: str, depth: int, sub_depth: int | None, *, epochs: int = 300, seed: int = 0
) -> tuple[float, float]:
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack(mode, depth, sub_depth)
    xg = x.clone().requires_grad_(True)
    net(xg).sum().backward()  # type: ignore[no-untyped-call]
    assert xg.grad is not None
    gnorm = float(xg.grad.norm() / xg.shape[0] ** 0.5)
    net = build_residual_stack(mode, depth, sub_depth)  # fresh for training
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    mse = _CAP if (not np.isfinite(loss_val) or loss_val > _CAP) else loss_val
    return mse, min(gnorm, _CAP)


def main() -> None:
    """Run the sweep and write the committed results JSON."""
    rows: list[dict[str, float | str | int]] = []
    for mode in _MODES:
        for depth in _DEPTHS:
            for k in _KS:
                if k is not None and k > depth:
                    continue
                mse, gnorm = _run(mode, depth, k)
                rows.append(
                    {
                        "mode": mode,
                        "depth": depth,
                        "skip_k": "plain" if k is None else k,
                        "final_train_mse": round(mse, 4),
                        "init_grad_norm": float(f"{gnorm:.4g}"),
                    }
                )
                print(f"{mode:9} d{depth:<2} K={k!s:5} mse={mse:.4f} g={gnorm:.3e}")  # noqa: T201
    out = (
        Path(__file__).resolve().parent
        / "results"
        / "deep-residual"
        / "trainability.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
