"""MonoResidual gate fix: sensitivity to un-standardized input scale.

Companion to `benchmarks/monoresidual_gate_ablation.py`. That script fixes
the input scale (`x ~ U(0, 1)`) and varies the two gate-fix levers; this one
fixes the levers to the recommended construction — A = near-zero F init
(`_NEAR_ZERO_SCALE = 1e-3`), B = the dead-zone-free `softplus` beta-gate —
and instead sweeps the *input scale* `s`, with `x ~ U(0, s)` and the teacher
target standardized as before.

The story: the near-zero-F fix keeps each block near-identity *at init*
regardless of `s` (`init_f_rms_last` stays tiny), but the `mixed`-mode
first layer and the near-open `softplus` gate (`g_beta(0) = softplus(0)
~= 0.69`, not 0) both scale with the raw input magnitude. So the last
block's total output RMS (`init_block_out_rms_last`) grows with `s`, and
training a 16-deep stack on unstandardized, large-scale inputs (`s=10, 100`)
starts far from the (always unit-scale) target and fails to converge in a
fixed step budget — motivating the docs' input-standardization requirement
(the fix is necessary but not sufficient without also standardizing inputs).

Run: ``uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_scale``

Pass ``--out PATH`` to write the per-scale rows as JSON (consumed by
``tests/benchmarks/test_monoresidual_gate_evidence.py`` and the docs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks.monoresidual_gate_ablation import _D, _W, _Block, _finite, _teacher
from mononet.torch import MonoLinear

_DEPTH = 16
_STEPS = 400
_SCALES: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)


def _run(scale: float, *, device: torch.device, seed: int = 0) -> dict[str, float]:
    """Build the A+B-fixed stack on `x ~ U(0, scale)`; report init RMS + train MSE.

    :param scale: Upper bound of the uniform input distribution.
    :param device: Torch device to run on.
    :param seed: Seed shared by torch and numpy for reproducibility.
    :returns: A row dict with keys ``scale``, ``init_f_rms_last``,
        ``init_block_out_rms_last``, ``train_mse``.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x_tr_np = rng.uniform(0, scale, (16000, _D))
    y_raw = _teacher(x_tr_np, np.random.default_rng(seed))
    mu, sd = y_raw.mean(), (y_raw.std() or 1.0)
    x_tr = torch.tensor(x_tr_np, dtype=torch.float32, device=device)
    y_tr = torch.tensor(
        (y_raw - mu) / sd, dtype=torch.float32, device=device
    ).unsqueeze(1)

    net = nn.Sequential(
        MonoLinear(_D, _W, mode="mixed", activation="elu"),
        *[_Block(a_mode="nearzero", softplus_gate=True) for _ in range(_DEPTH)],
        MonoLinear(_W, 1, mode="mixed"),
    ).to(device)
    blocks = [m for m in net if isinstance(m, _Block)]
    last = blocks[-1]

    with torch.no_grad():
        h = net[0](x_tr)  # blocks operate in hidden space, not raw input space
        for b in blocks[:-1]:
            h = b(h)
        f_last = last.f(h)
        block_out_last = last(h)
        init_f_rms = float(f_last.pow(2).mean().sqrt())
        init_block_out_rms = float(block_out_last.pow(2).mean().sqrt())

    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.MSELoss()
    loss = torch.zeros((), device=device)
    for _ in range(_STEPS):
        opt.zero_grad()
        loss = loss_fn(net(x_tr), y_tr)
        loss.backward()
        opt.step()

    return {
        "scale": scale,
        "init_f_rms_last": round(_finite(init_f_rms), 6),
        "init_block_out_rms_last": round(_finite(init_block_out_rms), 6),
        "train_mse": round(_finite(float(loss)), 6),
    }


def main(out: Path | None = None) -> None:
    """Sweep input scale on the A+B-fixed construction; print/write the table.

    :param out: When given, write the per-scale rows as JSON to this path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(  # noqa: T201
        "depth=16, mixed mode, A=nearzero B=softplus (the fix) | sweeping input scale s"
    )
    rows = []
    for scale in _SCALES:
        row = _run(scale, device=device)
        rows.append(row)
        print(  # noqa: T201
            f"s={scale:<6} init_f_rms_last={row['init_f_rms_last']:.4f} "
            f"init_block_out_rms_last={row['init_block_out_rms_last']:.4f} "
            f"train_mse={row['train_mse']:.4f}"
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Write rows as JSON.")
    args = parser.parse_args()
    main(args.out)
