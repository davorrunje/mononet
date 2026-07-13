"""MonoResidual gate ablation: why deep monotone-residual depth stays inert.

Evidence backing the skip-connection gate design (see the fix spec/plan). A deep
``absolute``-mode ``MonoResidual`` stack fails to use its depth because the
F-path gate ``g_beta`` (token ``scaled_elu``) sits in a **bootstrap trap**: at
init F is random (not identity), so engaging it *raises* loss, gradient descent
drives the gate parameter negative, and ``scaled_elu``'s negative-side gradient
dead-zone pins it there. ``g_beta`` stays ~0 forever and the "deep" net is really
a scaled identity chain.

This script isolates the two candidate levers on a synthetic monotone
ReLU-teacher target (self-contained, no dataset loaders):

* ``A`` — zero-init F's last layer, so each block is an **exact identity** at
  init (no harmful push on the gate parameter);
* ``B`` — swap the dead-zone gate ``scaled_elu`` for a dead-zone-free positive
  gate (``softplus``), preserving monotonicity (``g_beta >= 0``).

Result (depth 16): baseline stays trapped (``g_beta==0``); ``A`` alone escapes
the trap; ``B`` alone **diverges** (a dead-zone-free gate engages a *random* F
through every block → exponential blow-up), proving the gate activation cannot
be swapped in isolation; ``A+B`` is best. Conclusion: ``A`` (identity-at-init) is
the necessary safety property; ``B`` helps only in conjunction with it.

Run: ``uv run --extra torch --group bench python benchmarks/monoresidual_gate_ablation.py``
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from mononet.torch import MonoLinear

_D, _W = 6, 32
_GATE_EPS = 0.001


def _teacher(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Build a monotone target: non-negative weights + relu + non-negative skips."""
    h = x @ rng.uniform(0, 1, (_D, 8))
    for _ in range(4):
        h = np.maximum(h @ rng.uniform(0, 1, (8, 8)) + rng.uniform(-0.5, 0.5, 8), 0.0)
        h = h + h @ rng.uniform(0, 1, (8, 8))
    return (h @ rng.uniform(0, 1, (8, 1)))[:, 0]


class _Block(nn.Module):
    """One residual block, with the two ablation levers as flags."""

    def __init__(self, *, zero_init_f: bool, softplus_gate: bool) -> None:
        super().__init__()
        self.softplus_gate = softplus_gate
        self.f_in = MonoLinear(_W, _W, mode="absolute", activation="elu")
        self.f_out = MonoLinear(_W, _W, mode="absolute", activation="elu")
        if zero_init_f:
            with torch.no_grad():
                self.f_out.weight.zero_()
                if self.f_out.bias is not None:
                    self.f_out.bias.zero_()
        self.beta = nn.Parameter(torch.zeros(()))

    def g_beta(self) -> torch.Tensor:
        """Positive gate value; ``softplus`` (dead-zone-free) or ``scaled_elu``."""
        if self.softplus_gate:
            return functional.softplus(self.beta)
        pos = torch.clamp(self.beta, min=0)
        neg = _GATE_EPS * torch.exp(torch.clamp(self.beta, max=0) / _GATE_EPS)
        return pos + neg

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated residual: ``x + g_beta * F(x)``."""
        fx: torch.Tensor = self.f_out(self.f_in(x))
        out: torch.Tensor = x + self.g_beta() * fx
        return out


def _run(
    *,
    zero_init_f: bool,
    softplus_gate: bool,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    x_te: torch.Tensor,
    y_te: torch.Tensor,
    depth: int = 16,
    steps: int = 400,
) -> None:
    """Train one deep stack and print gate opening / loss / block-RMS."""
    net = nn.Sequential(
        MonoLinear(_D, _W, mode="absolute", activation="elu"),
        *[
            _Block(zero_init_f=zero_init_f, softplus_gate=softplus_gate)
            for _ in range(depth)
        ],
        MonoLinear(_W, 1, mode="absolute"),
    )
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    loss_fn = nn.MSELoss()
    rms_last = 0.0
    h = x_tr[:2048]
    with torch.no_grad():
        for module in net:
            h = module(h)
            if isinstance(module, _Block):
                rms_last = float(h.pow(2).mean().sqrt())
    loss = torch.zeros(())
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(net(x_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        test = float(loss_fn(net(x_te), y_te))
    gates = [float(m.g_beta()) for m in net if isinstance(m, _Block)]
    print(  # noqa: T201
        f"A={int(zero_init_f)} B={int(softplus_gate)}: "
        f"train {float(loss):.4f} test {test:.4f} | "
        f"g_beta[{min(gates):.3f},{max(gates):.3f}] | initRMS[last]={rms_last:.1f}"
    )


def main() -> None:
    """Run the four-config ablation and print the table."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x_tr_np = rng.uniform(0, 1, (16000, _D))
    y_raw = _teacher(x_tr_np, np.random.default_rng(0))
    mu, sd = y_raw.mean(), (y_raw.std() or 1.0)
    x_te_np = rng.uniform(0, 1, (4000, _D))
    x_tr = torch.tensor(x_tr_np, dtype=torch.float32)
    y_tr = torch.tensor((y_raw - mu) / sd, dtype=torch.float32).unsqueeze(1)
    x_te = torch.tensor(x_te_np, dtype=torch.float32)
    y_te = torch.tensor(
        (_teacher(x_te_np, np.random.default_rng(0)) - mu) / sd, dtype=torch.float32
    ).unsqueeze(1)

    print(  # noqa: T201
        "A=zero-init F | B=softplus gate | depth=16, monotone teacher target"
    )
    for zero_init_f, softplus_gate in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        _run(
            zero_init_f=zero_init_f,
            softplus_gate=softplus_gate,
            x_tr=x_tr,
            y_tr=y_tr,
            x_te=x_te,
            y_te=y_te,
        )


if __name__ == "__main__":
    main()
