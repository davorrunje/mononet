"""Smoke tests for the committed MonoResidual gate-fix evidence JSON.

Guards against the committed JSON under
``benchmarks/results/monoresidual-gate/`` silently drifting away from the
qualitative claims the docs render (see ``benchmarks/monoresidual_gate_*.py``
for how each file is produced).
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "monoresidual-gate"
)


def test_ablation_json_shows_fix_beats_traps() -> None:
    """The A+B fix opens the gate and trains F; exact-zero freezes F."""
    rows = json.loads((RESULTS / "ablation.json").read_text())
    by = {(r["a_mode"], r["gate"]): r for r in rows}
    fix = by[("nearzero", "softplus")]
    assert fix["g_beta_max"] > 0.1
    assert fix["f_moved"] == fix["n_blocks"]
    # exact-zero freezes F's weights regardless of the gate.
    assert by[("exactzero", "softplus")]["f_moved"] == 0


def test_trap_json_shows_closed_gate() -> None:
    """The pre-fix trap (scaled_elu gate, random F) pins g_beta near 0."""
    trap = json.loads((RESULTS / "trap.json").read_text())
    assert trap["final"]["g_beta_max"] < 0.05


def test_scale_json_shows_unit_scale_sensitivity() -> None:
    """The A+B fix trains at unit input scale but breaks at large scale."""
    rows = json.loads((RESULTS / "scale.json").read_text())
    by = {r["scale"]: r for r in rows}
    assert by[1.0]["train_mse"] < 0.1  # unit-scale inputs train
    assert by[100.0]["train_mse"] > 10.0  # large-scale inputs break
