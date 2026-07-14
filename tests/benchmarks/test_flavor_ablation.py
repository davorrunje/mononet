"""Smoke + enumeration tests for the flavor-ablation grid runner."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.flavor_ablation import ablation_cells, run_dataset_ablation

if TYPE_CHECKING:
    from pathlib import Path


def test_focused_cells_cover_all_flavors() -> None:
    cells = ablation_cells(focused=True)
    flavors = {(c.mode, c.alt_init) for c in cells}
    assert flavors == {
        ("mixed", None),
        ("split", None),
        ("alternate", "composition"),
        ("alternate", "legacy"),
    }
    assert {c.activation for c in cells} == {"relu", "elu", "softplus", "selu"}
    assert {c.depth for c in cells} == {4, 8, 16}
    assert len(cells) == 4 * 4 * 3


def test_smoke_run_writes_records_with_divergence(tmp_path: Path) -> None:
    # smoke: subsampled synthetic dataset, 1 seed, 2 epochs — asserts schema,
    # not science.
    out = run_dataset_ablation(
        "synth_lattice_clow", "torch", lr_sweep=False, out_dir=tmp_path, smoke=True
    )
    recs = json.loads(out.read_text())
    assert recs, "no records written"
    r = recs[0]
    for key in (
        "dataset",
        "mode",
        "alt_init",
        "activation",
        "depth",
        "lr",
        "metric_iqm",
        "metric_iqr",
        "epochs_median",
        "divergence_rate",
        "collapsed",
        "n_seeds",
    ):
        assert key in r, f"missing {key}"
    assert r["dataset"] == "synth_lattice_clow"
    assert 0.0 <= r["divergence_rate"] <= 1.0


def test_lr_sweep_fixes_depth_eight(tmp_path: Path) -> None:
    out = run_dataset_ablation(
        "synth_lattice_clow", "torch", lr_sweep=True, out_dir=tmp_path, smoke=True
    )
    recs = json.loads(out.read_text())
    assert {r["depth"] for r in recs} == {8}
    assert {r["lr"] for r in recs} == {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}
