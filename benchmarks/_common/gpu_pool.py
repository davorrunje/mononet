"""Shared work-stealing GPU subprocess pool for the benchmark launchers.

Runs one subprocess per work item across a pool of devices: each item is
pinned to a device via ``$MONONET_TORCH_DEVICE``, and items are dispatched to
whichever device frees up next (a work-stealing queue). This is the one place
the launchers' fan-out lives — :mod:`benchmarks.stage2_launch` and
:mod:`benchmarks.sensitivity` both call it instead of re-implementing the
pool. Concurrency comes only from running multiple single-threaded subprocesses
across the device pool (threaded Optuna deadlocks under process/thread nesting
— see `benchmarks.stage2_launch`), so every command a caller builds must be
single-threaded.
"""

from __future__ import annotations

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

T = TypeVar("T")


def fan_out(
    items: Sequence[T],
    devices: Sequence[str],
    cmd_builder: Callable[[T, str], list[str]],
    *,
    label: Callable[[T], str] | None = None,
) -> list[T]:
    """Run one subprocess per item across a work-stealing device pool.

    `devices` encodes per-GPU concurrency: repeat a device to run more items on
    it at once (e.g. ``["cuda:0", "cuda:1", "cuda:0", "cuda:1"]`` = 2 per GPU).
    Each subprocess runs with ``$MONONET_TORCH_DEVICE`` set to its device and
    ``check=True`` (a failing item raises `subprocess.CalledProcessError`).

    :param items: Work items, one subprocess each.
    :param devices: Device pool; one slot per concurrent subprocess.
    :param cmd_builder: Builds the argv for an item on a device:
        ``cmd_builder(item, device) -> list[str]``.
    :param label: Optional short label for an item, used in the start/done log
        lines; defaults to ``str(item)``.
    :returns: The items, in completion order.
    """
    dev_q: Queue[str] = Queue()
    for d in devices:
        dev_q.put(d)

    def _task(item: T) -> T:
        device = dev_q.get()
        tag = label(item) if label else str(item)
        t0 = time.monotonic()
        print(f"[start] {tag} -> {device}", flush=True)  # noqa: T201
        try:
            subprocess.run(
                cmd_builder(item, device),
                env={**os.environ, "MONONET_TORCH_DEVICE": device},
                check=True,
            )
            return item
        finally:
            dev_q.put(device)
            print(  # noqa: T201
                f"[done ] {tag} ({device}) {time.monotonic() - t0:.0f}s",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(_task, it) for it in items]
        return [f.result() for f in as_completed(futs)]
