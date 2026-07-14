r"""Parallel, multi-GPU launcher for the flavor-ablation grid.

Runs each dataset's :mod:`benchmarks.flavor_ablation` invocation as its own
subprocess pinned to a GPU via ``$MONONET_TORCH_DEVICE``. Concurrency comes
only from running multiple dataset subprocesses across the device pool; each
subprocess writes its own ``<dataset>[-lrsweep].json`` under ``--out-dir``,
so there is no merge step.

Run the focused sweep on both GPUs::

    uv run --group bench python -m benchmarks.flavor_ablation_launch \\
        --datasets heart,auto,synth_lattice_clow,synth_lattice_cmid,synth_lattice_chigh \\
        --backend torch --devices cuda:0,cuda:1 \\
        --out-dir benchmarks/results/flavor-ablation

``--dry-run`` prints the (dataset -> device) plan and the exact commands
without spawning anything.
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

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "flavor-ablation"


def build_command(
    dataset: str, backend: str, out_dir: Path, *, lr_sweep: bool
) -> list[str]:
    """Build the ``benchmarks.flavor_ablation`` subprocess argv for one dataset.

    :param dataset: Dataset name, forwarded as ``--dataset``.
    :param backend: Backend, forwarded as ``--backend``.
    :param out_dir: Forwarded as ``--out-dir``.
    :param lr_sweep: If True, append ``--lr-sweep``.
    :returns: The argv list.
    """
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.flavor_ablation",
        "--dataset",
        dataset,
        "--backend",
        backend,
        "--out-dir",
        str(out_dir),
    ]
    if lr_sweep:
        cmd.append("--lr-sweep")
    return cmd


def plan_assignments(datasets: list[str], devices: list[str]) -> list[tuple[str, str]]:
    """Round-robin datasets over the device pool (the dry-run/static plan).

    The live :func:`run_parallel` uses a work-stealing queue; this static plan
    is what the dry-run reports and what the mapping converges to when every
    subprocess takes roughly the same time.

    :param datasets: Dataset names.
    :param devices: Device pool.
    :returns: ``(dataset, device)`` pairs in round-robin order.
    """
    return [(d, devices[i % len(devices)]) for i, d in enumerate(datasets)]


def _run_dataset(
    dataset: str, device: str, backend: str, out_dir: Path, *, lr_sweep: bool
) -> str:
    """Run one dataset's ablation as a subprocess pinned to ``device``.

    :param dataset: Dataset name.
    :param device: Torch device string set as ``$MONONET_TORCH_DEVICE``.
    :param backend: Backend name.
    :param out_dir: Forwarded as ``--out-dir``.
    :param lr_sweep: Forwarded as ``--lr-sweep`` if True.
    :returns: ``dataset``, for the caller to track completion.
    """
    env = {**os.environ, "MONONET_TORCH_DEVICE": device}
    subprocess.run(
        build_command(dataset, backend, out_dir, lr_sweep=lr_sweep),
        env=env,
        check=True,
    )
    return dataset


def run_parallel(
    *,
    datasets: tuple[str, ...],
    devices: list[str],
    backend: str,
    out_dir: Path,
    lr_sweep: bool = False,
    dry_run: bool = False,
) -> list[tuple[str, str]] | list[str]:
    """Run the ablation for all ``datasets``, one subprocess per dataset.

    :param datasets: Dataset names, one subprocess each.
    :param devices: Device pool; one slot per concurrent subprocess (repeat a
        device to run more datasets on it at once).
    :param backend: Backend name forwarded to each subprocess.
    :param out_dir: Passed through to each subprocess as ``--out-dir``.
    :param lr_sweep: Passed through as ``--lr-sweep`` if True.
    :param dry_run: If True, print the plan + commands and return the static
        ``(dataset, device)`` assignment plan without spawning anything.
    :returns: In dry-run, the ``(dataset, device)`` plan; otherwise the dataset
        names in completion order.
    """
    if dry_run:
        plan = plan_assignments(list(datasets), devices)
        for dataset, device in plan:
            cmd = build_command(dataset, backend, out_dir, lr_sweep=lr_sweep)
            print(  # noqa: T201
                f"[plan] {dataset} -> {device}: "
                f"MONONET_TORCH_DEVICE={device} {' '.join(cmd)}",
                flush=True,
            )
        return plan

    out_dir.mkdir(parents=True, exist_ok=True)
    dev_q: Queue[str] = Queue()
    for d in devices:
        dev_q.put(d)

    def _task(dataset: str) -> str:
        device = dev_q.get()
        t0 = time.monotonic()
        print(f"[start] dataset={dataset} -> {device}", flush=True)  # noqa: T201
        try:
            return _run_dataset(dataset, device, backend, out_dir, lr_sweep=lr_sweep)
        finally:
            dev_q.put(device)
            print(  # noqa: T201
                f"[done ] dataset={dataset} ({device}) {time.monotonic() - t0:.0f}s",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        futs = [ex.submit(_task, name) for name in datasets]
        done = [f.result() for f in as_completed(futs)]
    print(f"finished {len(done)} datasets -> {out_dir}", flush=True)  # noqa: T201
    return done


def main() -> None:
    """CLI entry: distribute the flavor-ablation grid across GPUs."""
    ap = argparse.ArgumentParser(description="parallel multi-GPU flavor ablation")
    ap.add_argument("--datasets", required=True, help="comma-separated dataset names")
    ap.add_argument("--backend", default="torch", choices=("torch", "jax", "keras"))
    ap.add_argument(
        "--devices",
        default="cuda:0,cuda:1",
        help="comma-separated device slots; repeat a device for more concurrency",
    )
    ap.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT_DIR)
    ap.add_argument("--lr-sweep", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    run_parallel(
        datasets=tuple(args.datasets.split(",")),
        devices=args.devices.split(","),
        backend=args.backend,
        out_dir=args.out_dir,
        lr_sweep=args.lr_sweep,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
