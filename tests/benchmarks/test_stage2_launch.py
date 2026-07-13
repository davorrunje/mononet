from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks.stage2_launch import run_parallel

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_run_parallel_distributes_datasets_across_the_device_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each dataset runs once, on a pool device, with both devices used.

    4 datasets over a 2-device pool. Distribution is load-balanced by the
    work-stealing queue (like ``screen_launch``), so the exact per-device split
    is not fixed — only completeness, pool-membership, and both-devices-used
    are guaranteed.
    """
    calls: list[tuple[list[str], dict[str, str]]] = []

    def _fake_run(cmd: list[str], env: dict[str, str], check: bool) -> None:
        assert check is True
        calls.append((cmd, env))

    monkeypatch.setattr("benchmarks.stage2_launch.subprocess.run", _fake_run)

    datasets = ("auto", "heart", "compas", "loan")
    devices = ["cuda:0", "cuda:1"]
    done = run_parallel(
        datasets=datasets,
        devices=devices,
        out_dir=tmp_path,
    )

    assert sorted(done) == sorted(datasets)
    assert len(calls) == 4

    by_device: dict[str, list[str]] = {"cuda:0": [], "cuda:1": []}
    for cmd, env in calls:
        device = env["MONONET_TORCH_DEVICE"]
        name = cmd[cmd.index("--datasets") + 1]
        by_device[device].append(name)

    # completeness + pool-membership (indexing by_device would KeyError on an
    # out-of-pool device), and both devices are used — but NOT an exact split,
    # which the work-stealing queue does not guarantee.
    assert sorted(by_device["cuda:0"] + by_device["cuda:1"]) == sorted(datasets)
    assert len(by_device["cuda:0"]) >= 1
    assert len(by_device["cuda:1"]) >= 1


def test_run_parallel_passes_n_jobs_one_and_pinned_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every subprocess command hardcodes --n-jobs 1 and a valid pool device."""
    calls: list[tuple[list[str], dict[str, str]]] = []

    def _fake_run(cmd: list[str], env: dict[str, str], check: bool) -> None:
        calls.append((cmd, env))

    monkeypatch.setattr("benchmarks.stage2_launch.subprocess.run", _fake_run)

    devices = ["cuda:0", "cuda:1"]
    storage_dir = tmp_path / "studies"
    run_parallel(
        datasets=("adult", "taiwan"),
        devices=devices,
        out_dir=tmp_path,
        storage_dir=storage_dir,
    )

    assert len(calls) == 2
    for cmd, env in calls:
        assert "--n-jobs" in cmd
        assert cmd[cmd.index("--n-jobs") + 1] == "1"
        assert env["MONONET_TORCH_DEVICE"] in devices
        assert "--out-dir" in cmd
        assert cmd[cmd.index("--out-dir") + 1] == str(tmp_path)
        assert "--storage-dir" in cmd
        assert cmd[cmd.index("--storage-dir") + 1] == str(storage_dir)


def test_run_parallel_omits_storage_dir_when_not_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """storage_dir is optional; omitted entirely when not passed."""
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], env: dict[str, Any], check: bool) -> None:
        calls.append(cmd)

    monkeypatch.setattr("benchmarks.stage2_launch.subprocess.run", _fake_run)

    run_parallel(datasets=("auto",), devices=["cuda:0"], out_dir=tmp_path)

    assert len(calls) == 1
    assert "--storage-dir" not in calls[0]
