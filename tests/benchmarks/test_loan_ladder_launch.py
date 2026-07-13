from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.loan_ladder_launch import merge_partials, run_parallel

if TYPE_CHECKING:
    import pytest


def test_merge_partials_concatenates_and_sorts(tmp_path: Path) -> None:
    """merge_partials concatenates per-cell record lists, ordered by (n, arm)."""
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps([{"n": 400, "arm": "deep", "test_iqm": 0.60}]))
    p2 = tmp_path / "b.json"
    p2.write_text(
        json.dumps(
            [
                {"n": 100, "arm": "shallow", "test_iqm": 0.50},
                {"n": 100, "arm": "deep", "test_iqm": 0.55},
            ]
        )
    )
    recs = merge_partials([p1, p2])
    assert [(r["n"], r["arm"]) for r in recs] == [
        (100, "deep"),
        (100, "shallow"),
        (400, "deep"),
    ]


def test_run_parallel_threads_dataset_into_subprocess_cmd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--dataset lc is threaded into every cell's subprocess command (no real training)."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, env: Any, check: bool) -> None:
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps([{"n": 100, "arm": "shallow", "test_iqm": 0.5}]))

    monkeypatch.setattr("benchmarks.loan_ladder_launch.subprocess.run", fake_run)

    out = tmp_path / "lc.json"
    run_parallel(
        ns=(100,),
        arms=("shallow",),
        devices=["cpu"],
        budget={
            "n_trials": 1,
            "search_seeds": 1,
            "final_seeds": 1,
            "epochs": 1,
            "n_jobs": 1,
        },
        out=out,
        tmpdir=tmp_path / "_partial",
        dataset="lc",
    )

    assert calls
    for cmd in calls:
        assert cmd[cmd.index("--dataset") + 1] == "lc"
    assert out.exists()


def test_run_parallel_defaults_dataset_to_loan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Omitting --dataset keeps the loan back-compat default."""
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], *, env: Any, check: bool) -> None:
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.write_text(json.dumps([]))

    monkeypatch.setattr("benchmarks.loan_ladder_launch.subprocess.run", fake_run)

    run_parallel(
        ns=(100,),
        arms=("shallow",),
        devices=["cpu"],
        budget={
            "n_trials": 1,
            "search_seeds": 1,
            "final_seeds": 1,
            "epochs": 1,
            "n_jobs": 1,
        },
        out=tmp_path / "loan.json",
        tmpdir=tmp_path / "_partial",
    )

    assert calls
    for cmd in calls:
        assert cmd[cmd.index("--dataset") + 1] == "loan"
