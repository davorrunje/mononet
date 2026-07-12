r"""Parallel, multi-GPU launcher for the multi-dataset large-screen run.

Runs each dataset's :func:`benchmarks.large_screen_run.screen_dataset` as its
own subprocess pinned to a GPU via ``$MONONET_TORCH_DEVICE``. Concurrency comes
*only* from running multiple dataset subprocesses across the device pool, each
single-threaded (``--n-jobs 1``, hardcoded below): threaded Optuna
(``n_jobs > 1``) deadlocks under this launcher's process/thread nesting — we
hit that deadlock twice during development, so this is a hard constraint, not
a tuning knob. Each subprocess writes its own per-dataset record JSON;
:func:`merge_screens` assembles the committed ``results/screen/all.json``.

Run on both GPUs (5090 + 3090), datasets distributed across the pool:
``uv run --group bench python -m benchmarks.screen_launch \\
    --datasets adult,taiwan,polish,german,lc --devices cuda:0,cuda:1``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from typing import Any

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "screen"


def merge_screens(paths: list[Path]) -> list[dict[str, Any]]:
    """Read per-dataset screen-record JSON files and sort by ``name``.

    :param paths: Per-dataset screen record files, each a single JSON object
        with the schema :func:`benchmarks.large_screen_run.screen_dataset`
        emits (``name``, ``n_full``, ``deep_iqm``, ``shallow_iqm``, ``delta``,
        ``delta_lo``, ``delta_hi``, ``margin``, ``verdict``).
    :returns: All records, ordered by ascending ``name``.
    """
    records: list[dict[str, Any]] = [json.loads(Path(p).read_text()) for p in paths]
    records.sort(key=lambda r: str(r["name"]))
    return records


def _run_dataset(name: str, device: str, out: Path, budget: dict[str, int]) -> Path:
    """Run one dataset's screen as a subprocess pinned to ``device``.

    Always passes ``--n-jobs 1`` to the subprocess: multi-dataset concurrency
    comes from the device pool (multiple processes), not from threaded Optuna
    inside a single process, which deadlocks (see module docstring).
    """
    env = {**os.environ, "MONONET_TORCH_DEVICE": device}
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.large_screen_run",
        "--dataset",
        name,
        "--out",
        str(out),
        "--n-trials",
        str(budget["n_trials"]),
        "--search-seeds",
        str(budget["search_seeds"]),
        "--final-seeds",
        str(budget["final_seeds"]),
        "--epochs",
        str(budget["epochs"]),
        "--n-jobs",
        "1",
    ]
    subprocess.run(cmd, env=env, check=True)
    return out


def run_parallel(
    *,
    datasets: tuple[str, ...],
    devices: list[str],
    budget: dict[str, int],
    out_dir: Path,
) -> Path:
    """Screen all `datasets` across `devices` and merge into ``<out_dir>/all.json``.

    `devices` encodes per-GPU concurrency: repeat a device to run more
    datasets on it at once (e.g. ``["cuda:0", "cuda:1", "cuda:0", "cuda:1"]`` =
    2 per GPU).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_q: Queue[str] = Queue()
    for d in devices:
        dev_q.put(d)

    def _task(name: str) -> Path:
        device = dev_q.get()
        t0 = time.monotonic()
        print(f"[start] dataset={name} -> {device}", flush=True)  # noqa: T201
        try:
            return _run_dataset(name, device, out_dir / f"{name}.json", budget)
        finally:
            dev_q.put(device)
            print(  # noqa: T201
                f"[done ] dataset={name} ({device}) {time.monotonic() - t0:.0f}s",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(_task, name) for name in datasets]
        per_dataset = [f.result() for f in as_completed(futs)]

    out = out_dir / "all.json"
    out.write_text(json.dumps(merge_screens(sorted(per_dataset)), indent=2) + "\n")
    print(f"merged {len(per_dataset)} datasets -> {out}", flush=True)  # noqa: T201
    return out


def main() -> None:
    """CLI entry: distribute the multi-dataset screen across GPUs and merge."""
    ap = argparse.ArgumentParser(description="parallel multi-GPU large-dataset screen")
    ap.add_argument("--datasets", required=True, help="comma-separated dataset names")
    ap.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="comma-separated device slots; repeat a device for more concurrency",
    )
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    args = ap.parse_args()

    budget = {
        "n_trials": args.n_trials,
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "epochs": args.epochs,
    }
    run_parallel(
        datasets=tuple(args.datasets.split(",")),
        devices=args.devices.split(","),
        budget=budget,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
