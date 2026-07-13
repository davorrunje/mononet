"""MonoResidual gate ablation: why deep monotone-residual depth stays inert.

Evidence backing the skip-connection gate design (see the fix spec/plan). A deep
``absolute``-mode ``MonoResidual`` stack fails to use its depth because the
F-path gate ``g_beta`` (token ``scaled_elu``) sits in a **bootstrap trap**: at
init F is random (not near-identity), so engaging it *raises* loss, gradient
descent drives the gate parameter negative, and ``scaled_elu``'s negative-side
gradient dead-zone pins it there. ``g_beta`` stays ~0 forever and the "deep" net
is really a scaled identity chain.

This script isolates two levers on a synthetic monotone teacher target
(self-contained, no dataset loaders):

* ``A`` — the F-path init. ``off`` = normal init (random F); ``exactzero`` =
  zero F's last layer (naive Fixup); ``nearzero`` = scale F's last-layer weight
  by ``_NEAR_ZERO_SCALE`` (near-identity, but nonzero).
* ``B`` — the F gate. ``scaled_elu`` (dead-zone) vs ``softplus`` (dead-zone-free,
  monotone).

Key subtlety: under the ``absolute`` construction ``F`` uses ``|W|``, whose
gradient at ``W=0`` is ``sign(0)=0`` — so **exact-zero init is a gradient fixed
point**: the last-layer weights never move and ``F`` degenerates to a per-block
learned *constant*, not an ``x``-dependent depth function. ``nearzero`` keeps the
weights trainable while still starting ``F ~= 0``. So the fix is near-zero init
(A) + softplus gate (B); exact-zero is itself a trap.

Run: ``uv run --extra torch --group bench python benchmarks/monoresidual_gate_ablation.py``

Pass ``--out PATH`` to additionally write the six rows as JSON (consumed by
``tests/benchmarks/test_monoresidual_gate_evidence.py`` and the docs); the
default (no ``--out``) invocation keeps printing only, as before.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from mononet.torch import MonoLinear

_D, _W = 6, 32
_GATE_EPS = 0.001
_NEAR_ZERO_SCALE = 1e-3
_MSE_CAP = 1.0e6


def _finite(value: float, cap: float = _MSE_CAP) -> float:
    """Clamp non-finite or blown-up values to a large sentinel for JSON safety.

    :param value: Raw scalar (may be ``nan``/``inf`` on a diverging run).
    :param cap: Sentinel magnitude substituted for non-finite/over-cap values.
    :returns: ``value`` if finite and within ``cap``, else ``cap``.
    """
    return cap if (not math.isfinite(value) or abs(value) > cap) else value


def _teacher(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build a monotone target: non-negative weights + relu + non-negative skips."""
    h = x @ rng.uniform(0, 1, (_D, 8))
    for _ in range(4):
        h = np.maximum(h @ rng.uniform(0, 1, (8, 8)) + rng.uniform(-0.5, 0.5, 8), 0.0)
        h = h + h @ rng.uniform(0, 1, (8, 8))
    return (h @ rng.uniform(0, 1, (8, 1)))[:, 0]


class _Block(nn.Module):
    """One residual block, parametrized by the two ablation levers."""

    def __init__(self, *, a_mode: str, softplus_gate: bool) -> None:
        super().__init__()
        self.softplus_gate = softplus_gate
        self.f_in = MonoLinear(_W, _W, mode="absolute", activation="elu")
        self.f_out = MonoLinear(_W, _W, mode="absolute", activation="elu")
        with torch.no_grad():
            if a_mode == "exactzero":
                self.f_out.weight.zero_()
            elif a_mode == "nearzero":
                self.f_out.weight.mul_(_NEAR_ZERO_SCALE)
            if a_mode in ("exactzero", "nearzero") and self.f_out.bias is not None:
                self.f_out.bias.zero_()
        self.beta = nn.Parameter(torch.zeros(()))

    def g_beta(self) -> torch.Tensor:
        """Positive gate value; ``softplus`` (dead-zone-free) or ``scaled_elu``."""
        if self.softplus_gate:
            return functional.softplus(self.beta)
        pos = torch.clamp(self.beta, min=0)
        neg = _GATE_EPS * torch.exp(torch.clamp(self.beta, max=0) / _GATE_EPS)
        return pos + neg

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the residual branch ``F(x)``."""
        out: torch.Tensor = self.f_out(self.f_in(x))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated residual: ``x + g_beta * F(x)``."""
        out: torch.Tensor = x + self.g_beta() * self.f(x)
        return out


def _run(
    *,
    a_mode: str,
    softplus_gate: bool,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    device: torch.device,
    depth: int = 16,
    steps: int = 400,
) -> dict[str, float | str | int]:
    """Train one deep stack; print and return gate/loss/F-weight-movement summary.

    :returns: A row dict with keys ``a_mode``, ``gate``, ``train``, ``test``,
        ``g_beta_min``, ``g_beta_max``, ``f_moved``, ``n_blocks``.
    """
    net = nn.Sequential(
        MonoLinear(_D, _W, mode="absolute", activation="elu"),
        *[_Block(a_mode=a_mode, softplus_gate=softplus_gate) for _ in range(depth)],
        MonoLinear(_W, 1, mode="absolute"),
    ).to(device)
    blocks = [m for m in net if isinstance(m, _Block)]
    w0 = [float(b.f_out.weight.detach().abs().sum()) for b in blocks]
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.MSELoss()
    loss = torch.zeros(())
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(x_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test = float(loss_fn(net(x_te), y_te))
        gates = [float(b.g_beta()) for b in blocks]
    moved = sum(
        1
        for b, w in zip(blocks, w0, strict=True)
        if abs(float(b.f_out.weight.detach().abs().sum()) - w) > 1e-8
    )
    label = f"A={a_mode:<9} B={'softplus' if softplus_gate else 'scaled_elu':<10}"
    print(  # noqa: T201
        f"{label}: train {float(loss):.4f} test {test:.4f} | "
        f"g_beta[{min(gates):.3f},{max(gates):.3f}] | "
        f"F-weights-moved {moved}/{len(blocks)}"
    )
    return {
        "a_mode": a_mode,
        "gate": "softplus" if softplus_gate else "scaled_elu",
        "train": round(_finite(float(loss)), 6),
        "test": round(_finite(test), 6),
        "g_beta_min": round(_finite(min(gates)), 6),
        "g_beta_max": round(_finite(max(gates)), 6),
        "f_moved": moved,
        "n_blocks": len(blocks),
    }


def main(out: Path | None = None) -> None:
    """Run the ablation grid, print the table, and optionally write JSON.

    :param out: When given, write the six result rows to this path as JSON.
        The default (`None`) keeps the original print-only behaviour.
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

    print(  # noqa: T201
        "depth=16, absolute mode, monotone teacher target | "
        f"near-zero scale={_NEAR_ZERO_SCALE}"
    )
    grid = [
        ("off", False),  # baseline: trapped
        ("exactzero", False),  # A exact-zero only: freezes F weights
        ("nearzero", False),  # A near-zero only: escapes trap, F trains
        ("off", True),  # B only: diverges
        ("exactzero", True),  # exact-zero + B: gate opens but F still frozen
        ("nearzero", True),  # near-zero + B: best
    ]
    rows = [
        _run(
            a_mode=a_mode,
            softplus_gate=softplus_gate,
            x_tr=x_tr,
            y_tr=y_tr,
            x_te=x_te,
            y_te=y_te,
            device=device,
        )
        for a_mode, softplus_gate in grid
    ]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None, help="Write rows as JSON.")
    args = parser.parse_args()
    main(args.out)
