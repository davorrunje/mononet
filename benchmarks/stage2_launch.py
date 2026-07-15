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
import sys
from pathlib import Path

from benchmarks._common.gpu_pool import fan_out

_DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results" / "phase2"


def run_parallel(
    *,
    datasets: tuple[str, ...],
    devices: list[str],
    out_dir: Path,
    storage_dir: Path | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """Run the Stage-A search for all `datasets`, one subprocess per dataset.

    Fans out over the device pool via
    :func:`benchmarks._common.gpu_pool.fan_out`. `devices` encodes per-GPU
    concurrency: repeat a device to run more datasets on it at once (e.g.
    ``["cuda:0", "cuda:1", "cuda:0", "cuda:1"]`` = 2 per GPU). Each subprocess
    is single-threaded (``--n-jobs 1``, hardcoded below): threaded Optuna
    (``n_jobs > 1``) deadlocks under this launcher's process/thread nesting.

    :param datasets: Dataset names to run, one subprocess each.
    :param devices: Device pool; one slot per concurrent subprocess.
    :param out_dir: Passed through to each subprocess as ``--out-dir``.
    :param storage_dir: Passed through to each subprocess as ``--storage-dir``
        if given.
    :param extra: Additional CLI args forwarded verbatim to each subprocess
        (e.g. ``--flavors``, ``--search-activation``, ``--max-depth``,
        ``--embed-layers``).
    :returns: Dataset names, in completion order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = extra or []

    def _cmd(name: str, device: str) -> list[str]:
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
        return cmd + extra

    done = fan_out(datasets, devices, _cmd, label=lambda n: f"dataset={n}")
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
    ap.add_argument(
        "--flavors", default=None, help="forwarded to benchmarks.search --flavors"
    )
    ap.add_argument(
        "--search-activation",
        action="store_true",
        help="forwarded to benchmarks.search --search-activation",
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="forwarded to benchmarks.search --max-depth",
    )
    ap.add_argument(
        "--embed-layers",
        type=int,
        default=None,
        help="forwarded to benchmarks.search --embed-layers",
    )
    args = ap.parse_args()

    extra: list[str] = []
    if args.flavors is not None:
        extra += ["--flavors", args.flavors]
    if args.search_activation:
        extra += ["--search-activation"]
    if args.max_depth is not None:
        extra += ["--max-depth", str(args.max_depth)]
    if args.embed_layers is not None:
        extra += ["--embed-layers", str(args.embed_layers)]

    run_parallel(
        datasets=tuple(args.datasets.split(",")),
        devices=args.devices.split(","),
        out_dir=args.out_dir,
        storage_dir=args.storage_dir,
        extra=extra,
    )


if __name__ == "__main__":
    main()
