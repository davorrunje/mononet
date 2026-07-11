"""Parallel, multi-GPU launcher for the loan size-ladder run.

Runs each ``(N, arm)`` cell as its own process pinned to a GPU via
``$MONONET_TORCH_DEVICE``, several concurrent per GPU (the nets are tiny, so one
process barely uses a GPU). Each cell writes a partial JSON; :func:`merge_partials`
assembles the committed ``results/size-ladder/loan.json``.

Run on both GPUs (5090 + 3090):
``uv run --group bench python -m benchmarks.loan_ladder_launch``
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

_LADDER = (5_000, 15_000, 45_000, 135_000, 1_000_000_000)  # last = full split
_ARMS = ("deep", "shallow")
_DEFAULT_OUT = Path(__file__).resolve().parent / "results" / "size-ladder" / "loan.json"


def merge_partials(paths: list[Path]) -> list[dict[str, Any]]:
    """Concatenate per-cell partial JSON record lists, sorted by ``(n, arm)``.

    :param paths: Partial JSON files, each a list of records.
    :returns: All records, ordered by ascending ``n`` then ``arm``.
    """
    records: list[dict[str, Any]] = []
    for p in paths:
        records.extend(json.loads(Path(p).read_text()))
    records.sort(key=lambda r: (int(r["n"]), str(r["arm"])))
    return records


def _run_cell(n: int, arm: str, device: str, out: Path, budget: dict[str, int]) -> Path:
    """Run one ``(N, arm)`` cell as a subprocess pinned to ``device``."""
    env = {**os.environ, "MONONET_TORCH_DEVICE": device}
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.loan_size_ladder_run",
        "--ns",
        str(n),
        "--arms",
        arm,
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
        str(budget["n_jobs"]),
    ]
    subprocess.run(cmd, env=env, check=True)
    return out


def run_parallel(
    *,
    ns: tuple[int, ...],
    arms: tuple[str, ...],
    devices: list[str],
    budget: dict[str, int],
    out: Path,
    tmpdir: Path,
) -> Path:
    """Run all ``(N, arm)`` cells across `devices` and merge into `out`.

    `devices` encodes per-GPU concurrency: repeat a device to run more cells on
    it at once (e.g. ``["cuda:0", "cuda:1", "cuda:0", "cuda:1"]`` = 2 per GPU).
    Heaviest cells are dispatched first so long poles start early.
    """
    tmpdir.mkdir(parents=True, exist_ok=True)
    # heavy-first: deep before shallow, larger N before smaller.
    jobs = sorted(
        ((n, arm) for n in ns for arm in arms),
        key=lambda j: (j[1] != "deep", -j[0]),
    )
    dev_q: Queue[str] = Queue()
    for d in devices:
        dev_q.put(d)

    def _task(job: tuple[int, str]) -> Path:
        n, arm = job
        device = dev_q.get()
        t0 = time.monotonic()
        print(f"[start] N={n} arm={arm} -> {device}", flush=True)  # noqa: T201
        try:
            return _run_cell(n, arm, device, tmpdir / f"loan-{n}-{arm}.json", budget)
        finally:
            dev_q.put(device)
            print(  # noqa: T201
                f"[done ] N={n} arm={arm} ({device}) {time.monotonic() - t0:.0f}s",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(_task, j) for j in jobs]
        partials = [f.result() for f in as_completed(futs)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merge_partials(sorted(partials)), indent=2) + "\n")
    print(f"merged {len(partials)} partials -> {out}", flush=True)  # noqa: T201
    return out


def main() -> None:
    """CLI entry: distribute the ladder across GPUs and merge the results."""
    ap = argparse.ArgumentParser(description="parallel multi-GPU loan size-ladder")
    ap.add_argument("--ns", default=",".join(str(n) for n in _LADDER))
    ap.add_argument("--arms", default=",".join(_ARMS))
    ap.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:0,cuda:1",
        help="comma-separated device slots; repeat a device for more concurrency",
    )
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    ap.add_argument("--tmpdir", type=Path, default=_DEFAULT_OUT.parent / "_partial")
    args = ap.parse_args()

    budget = {
        "n_trials": args.n_trials,
        "search_seeds": args.search_seeds,
        "final_seeds": args.final_seeds,
        "epochs": args.epochs,
        "n_jobs": args.n_jobs,
    }
    run_parallel(
        ns=tuple(int(x) for x in args.ns.split(",")),
        arms=tuple(args.arms.split(",")),
        devices=args.devices.split(","),
        budget=budget,
        out=args.out,
        tmpdir=args.tmpdir,
    )


if __name__ == "__main__":
    main()
