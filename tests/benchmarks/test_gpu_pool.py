from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks._common.gpu_pool import fan_out

if TYPE_CHECKING:
    import pytest


def test_fan_out_runs_every_item_once_on_a_pool_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], str]] = []

    def _fake_run(cmd: list[str], env: dict[str, str], check: bool) -> None:
        assert check is True
        calls.append((cmd, env["MONONET_TORCH_DEVICE"]))

    monkeypatch.setattr("benchmarks._common.gpu_pool.subprocess.run", _fake_run)

    items = ["a", "b", "c", "d"]
    devices = ["cuda:0", "cuda:1"]
    done = fan_out(items, devices, lambda it, dev: ["run", it, "--dev", dev])

    assert sorted(done) == sorted(items)
    assert len(calls) == 4
    # every item's argv was built by cmd_builder; every device is in the pool.
    ran = {cmd[1] for cmd, _ in calls}
    assert ran == set(items)
    assert all(dev in devices for _, dev in calls)
    assert {dev for _, dev in calls} == set(devices)  # both devices used
