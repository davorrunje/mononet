"""Deep-init trainability sweep (torch).

Trains plain `absolute`/`switch` `MonoLinear` stacks across depth on the synthetic
monotone target and records the final train MSE per (depth, method). Compares the new
default `absolute` init against the old `he_normal` default and `switch`. Writes
``benchmarks/results/deep-init/trainability.json`` (committed; read by
``docs/benchmarks/deep-init.ipynb``). Repo-only; never shipped in the wheel.

Run: ``uv run --extra torch --group bench python -m benchmarks.deep_init_run``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from benchmarks._common.init_diagnostics import synthetic_monotone
from mononet.torch import MonoLinear

if TYPE_CHECKING:
    from mononet.core.config import Mode

_DEPTHS: tuple[int, ...] = (2, 4, 8, 16)
# (label, mode, init)
_METHODS: tuple[tuple[str, Mode, str | None], ...] = (
    ("absolute (new init)", "absolute", None),
    ("absolute (he_normal)", "absolute", "he_normal"),
    ("switch", "switch", None),
)
_CAP = 1.0e6  # cap diverged losses so the committed JSON stays standard (no inf)


def _final_train_mse(
    mode: Mode,
    init: str | None,
    depth: int,
    *,
    epochs: int = 300,
    seed: int = 0,
    width: int = 32,
) -> float:
    """Train a depth-`depth` stack and return the final train MSE (capped at `_CAP`)."""
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    layers: list[nn.Module] = [
        MonoLinear(8, width, mode=mode, activation="elu", init=init)
    ]
    layers += [
        MonoLinear(width, width, mode=mode, activation="elu", init=init)
        for _ in range(depth - 1)
    ]
    layers.append(MonoLinear(width, 1, mode=mode, activation="elu", init=init))
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
    if not np.isfinite(loss_val) or loss_val > _CAP:
        return _CAP
    return loss_val


def main() -> None:
    """Run the sweep and write the committed results JSON."""
    rows: list[dict[str, float | str]] = []
    for depth in _DEPTHS:
        for label, mode, init in _METHODS:
            mse = _final_train_mse(mode, init, depth)
            rows.append(
                {"depth": depth, "method": label, "final_train_mse": round(mse, 4)}
            )
            print(f"depth {depth:2d}  {label:22s}  {mse:.4f}")  # noqa: T201
    out = (
        Path(__file__).resolve().parent / "results" / "deep-init" / "trainability.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
