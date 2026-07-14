"""MonoResidual gate trap: per-step instrumentation of the bootstrap trap.

Companion to `benchmarks/monoresidual_gate_ablation.py`. That script reports
only the *final* state of each ablation cell; this one instruments a single
cell — the pre-fix default (`A=off`, `B=scaled_elu`) — step by step, so the
trap's mechanics are visible over training rather than asserted from the
endpoint alone.

The construction mirrors the ablation's `("off", False)` row: a deep
`mixed`-mode stack of 16 residual blocks with random (not near-identity)
`F` and the dead-zone `scaled_elu` beta-gate, trained on the same synthetic
monotone teacher. At every step it records, aggregated over the 16 blocks:
`g_beta` min/max, raw `beta` min/max, mean block-output RMS (on a small fixed
diagnostic batch), and train/test MSE. The expected story: `g_beta` collapses
towards (and stays pinned near) 0 within the first few dozen steps — gradient
descent pushes `beta` negative once a random `F` is found to raise the loss,
and `scaled_elu`'s negative-side gradient dead-zone (`~eps * exp(beta/eps)`)
then prevents it from re-opening.

Run: ``uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_trap``

Pass ``--out PATH`` to write the per-step trace as JSON (consumed by
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
_DIAG_N = 256


def _block_out_rms(blocks: list[_Block], x: torch.Tensor) -> float:
    """Mean RMS of each block's output over a fixed diagnostic batch.

    :param blocks: The residual blocks, in forward order.
    :param x: Diagnostic input batch fed into the first block.
    :returns: The block-output RMS values, averaged over blocks.
    """
    with torch.no_grad():
        h = x
        rms = []
        for b in blocks:
            h = b(h)
            rms.append(float(h.pow(2).mean().sqrt()))
    return float(np.mean(rms))


def main(out: Path | None = None) -> None:
    """Train the pre-fix trap configuration and record per-step trap evidence.

    :param out: When given, write the per-step trace as JSON to this path.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x_tr_np = rng.uniform(0, 1, (16000, _D))
    y_raw = _teacher(x_tr_np, np.random.default_rng(0))
    mu, sd = y_raw.mean(), (y_raw.std() or 1.0)
    x_te_np = rng.uniform(0, 1, (4000, _D))
    x_tr = torch.tensor(x_tr_np, dtype=torch.float32, device=device)
    y_tr = torch.tensor(
        (y_raw - mu) / sd, dtype=torch.float32, device=device
    ).unsqueeze(1)
    x_te = torch.tensor(x_te_np, dtype=torch.float32, device=device)
    y_te = torch.tensor(
        (_teacher(x_te_np, np.random.default_rng(0)) - mu) / sd,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    # Pre-fix trap: random (non-near-zero) F + dead-zone scaled_elu gate.
    net = nn.Sequential(
        MonoLinear(_D, _W, mode="mixed", activation="elu"),
        *[_Block(a_mode="off", softplus_gate=False) for _ in range(_DEPTH)],
        MonoLinear(_W, 1, mode="mixed"),
    ).to(device)
    blocks = [m for m in net if isinstance(m, _Block)]
    input_layer = net[0]
    with torch.no_grad():
        x_diag = input_layer(x_tr[:_DIAG_N])  # blocks operate in hidden space
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.MSELoss()

    steps: list[dict[str, float | int]] = []
    for step in range(_STEPS):
        opt.zero_grad()
        loss = loss_fn(net(x_tr), y_tr)
        loss.backward()
        opt.step()
        # Downsample the trace: keep every 10th step plus the first and last,
        # so the JSON stays summary-sized while still showing g_beta collapse
        # within the first ~10 steps. The final row is always retained, so the
        # top-level ``final`` summary (asserted by the smoke test) is exact.
        if not (step % 10 == 0 or step == _STEPS - 1):
            continue
        with torch.no_grad():
            gates = [float(b.g_beta()) for b in blocks]
            betas = [float(b.beta) for b in blocks]
            test = float(loss_fn(net(x_te), y_te))
        steps.append(
            {
                "step": step,
                "train_mse": round(_finite(float(loss)), 6),
                "test_mse": round(_finite(test), 6),
                "g_beta_min": round(_finite(min(gates)), 6),
                "g_beta_max": round(_finite(max(gates)), 6),
                "beta_min": round(_finite(min(betas)), 6),
                "beta_max": round(_finite(max(betas)), 6),
                "block_out_rms": round(_finite(_block_out_rms(blocks, x_diag)), 6),
            }
        )

    result = {
        "config": {
            "depth": _DEPTH,
            "steps": _STEPS,
            "a_mode": "off",
            "gate": "scaled_elu",
        },
        "steps": steps,
        "final": steps[-1],
    }
    final = result["final"]
    assert isinstance(final, dict)
    print(  # noqa: T201
        f"trap final (step {final['step']}): "
        f"train {final['train_mse']:.4f} test {final['test_mse']:.4f} | "
        f"g_beta[{final['g_beta_min']:.4f},{final['g_beta_max']:.4f}]"
    )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Write trace as JSON.")
    args = parser.parse_args()
    main(args.out)
