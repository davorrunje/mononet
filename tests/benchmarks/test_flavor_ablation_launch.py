"""Launcher command-building + device round-robin (no subprocess spawn)."""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks.flavor_ablation_launch import (
    build_command,
    plan_assignments,
    run_parallel,
)


def test_build_command_shape() -> None:
    cmd = build_command("heart", "torch", Path("/out"), lr_sweep=False)
    assert cmd[:3] == [sys.executable, "-m", "benchmarks.flavor_ablation"]
    assert cmd[cmd.index("--dataset") + 1] == "heart"
    assert cmd[cmd.index("--backend") + 1] == "torch"
    assert cmd[cmd.index("--out-dir") + 1] == "/out"
    assert "--lr-sweep" not in cmd


def test_build_command_lr_sweep_flag() -> None:
    cmd = build_command("auto", "torch", Path("/out"), lr_sweep=True)
    assert "--lr-sweep" in cmd


def test_plan_assignments_round_robin() -> None:
    pairs = plan_assignments(["a", "b", "c", "d", "e"], ["cuda:0", "cuda:1"])
    assert pairs == [
        ("a", "cuda:0"),
        ("b", "cuda:1"),
        ("c", "cuda:0"),
        ("d", "cuda:1"),
        ("e", "cuda:0"),
    ]


def test_dry_run_spawns_nothing_and_returns_plan(tmp_path: Path) -> None:
    plan = run_parallel(
        datasets=("heart", "auto"),
        devices=["cuda:0", "cuda:1"],
        backend="torch",
        out_dir=tmp_path,
        lr_sweep=False,
        dry_run=True,
    )
    assert plan == [("heart", "cuda:0"), ("auto", "cuda:1")]
