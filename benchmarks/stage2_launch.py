r"""Parallel, multi-GPU launcher for the Stage-A flavor search.

Runs each dataset's :mod:`benchmarks.search` invocation as its own subprocess
pinned to a GPU via ``$MONONET_TORCH_DEVICE``. Concurrency comes *only* from
running multiple dataset subprocesses across the device pool, each
single-threaded (``--n-jobs 1``, hardcoded below): threaded Optuna
(``n_jobs > 1``) deadlocks under this launcher's process/thread nesting —
:mod:`benchmarks.screen_launch` hit that deadlock twice during development,
so this is a hard constraint, not a tuning knob. Each subprocess already
writes its own per-flavor result JSONs (see ``benchmarks.search``), so there
is no merge step here — this launcher only fans out and waits.

Run on both GPUs (5090 + 3090), datasets distributed across the pool::

    uv run --group bench python -m benchmarks.stage2_launch \\
        --datasets adult,taiwan,polish,german,lc --devices cuda:0,cuda:1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "phase2"


def _run_dataset(
    name: str,
    device: str,
    out_dir: Path,
    storage_dir: Path | None,
) -> str:
    """Run one dataset's flavor search as a subprocess pinned to ``device``.

    Always passes ``--n-jobs 1`` to the subprocess: multi-dataset concurrency
    comes from the device pool (multiple processes), not from threaded Optuna
    inside a single process, which deadlocks (see module docstring).

    :param name: Dataset name, forwarded as ``--datasets``.
    :param device: Torch device string set as ``$MONONET_TORCH_DEVICE`` in
        the subprocess environment.
    :param out_dir: Forwarded as ``--out-dir``; where per-flavor result JSONs
        land.
    :param storage_dir: Forwarded as ``--storage-dir`` if given; where the
        resumable Optuna study databases land.
    :returns: `name`, for the caller to track completion.
    """
    env = {**os.environ, "MONONET_TORCH_DEVICE": device}
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.search",
        "--datasets",
        name,
        "--n-jobs",
        "1",
        "--out-dir",
        str(out_dir),
    ]
    if storage_dir is not None:
        cmd += ["--storage-dir", str(storage_dir)]
    subprocess.run(cmd, env=env, check=True)
    return name


def run_parallel(
    *,
    datasets: tuple[str, ...],
    devices: list[str],
    out_dir: Path,
    storage_dir: Path | None = None,
) -> list[str]:
    """Run the Stage-A search for all `datasets`, one subprocess per dataset.

    `devices` encodes per-GPU concurrency: repeat a device to run more
    datasets on it at once (e.g. ``["cuda:0", "cuda:1", "cuda:0", "cuda:1"]``
    = 2 per GPU). Datasets are dispatched to whichever device frees up next
    (a work-stealing queue), so the mapping is round-robin only when every
    subprocess takes roughly the same time.

    :param datasets: Dataset names to run, one subprocess each.
    :param devices: Device pool; one slot per concurrent subprocess.
    :param out_dir: Passed through to each subprocess as ``--out-dir``.
    :param storage_dir: Passed through to each subprocess as ``--storage-dir``
        if given.
    :returns: Dataset names, in completion order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_q: Queue[str] = Queue()
    for d in devices:
        dev_q.put(d)

    def _task(name: str) -> str:
        device = dev_q.get()
        t0 = time.monotonic()
        print(f"[start] dataset={name} -> {device}", flush=True)  # noqa: T201
        try:
            return _run_dataset(name, device, out_dir, storage_dir)
        finally:
            dev_q.put(device)
            print(  # noqa: T201
                f"[done ] dataset={name} ({device}) {time.monotonic() - t0:.0f}s",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(_task, name) for name in datasets]
        done = [f.result() for f in as_completed(futs)]
    print(f"finished {len(done)} datasets -> {out_dir}", flush=True)  # noqa: T201
    return done


def main() -> None:
    """CLI entry: distribute the Stage-A search across GPUs."""
    ap = argparse.ArgumentParser(description="parallel multi-GPU Stage-A search")
    ap.add_argument("--datasets", required=True, help="comma-separated dataset names")
    ap.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="comma-separated device slots; repeat a device for more concurrency",
    )
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    ap.add_argument("--storage-dir", type=Path, default=None)
    args = ap.parse_args()

    run_parallel(
        datasets=tuple(args.datasets.split(",")),
        devices=args.devices.split(","),
        out_dir=args.out_dir,
        storage_dir=args.storage_dir,
    )


if __name__ == "__main__":
    main()
