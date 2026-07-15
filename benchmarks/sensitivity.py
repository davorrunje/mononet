"""CLI: reconstruct HP-search sensitivity curves from committed Optuna storage.

Two stages, so the expensive part parallelizes across GPUs like the base run:

- ``extract`` — for one dataset, read its committed study DBs and write a small
  JSON of per-flavor curves (Curve A best-so-far objective; Curve B
  test-of-incumbent via bounded `final_eval` re-eval) plus saturation rows.
  Reads storage only — it never re-runs the search.
- ``render`` — read the per-dataset JSONs and write the faceted figure
  (PNG + PDF) and the Markdown saturation table.
- ``run`` — fan ``extract`` out over a device pool
  (:func:`benchmarks._common.gpu_pool.fan_out`, the shared launcher pool), then
  ``render``. This is the one-shot entry that reproduces the committed figure::

      uv run --group bench python -m benchmarks.sensitivity run \
          --storage-dir benchmarks/results/alternate-base/studies \
          --curves-dir benchmarks/results/alternate-base/curves \
          --out docs/_static/hp-search-sensitivity \
          --devices cuda:0,cuda:1,cuda:0,cuda:1 --test-seeds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks._common.gpu_pool import fan_out
from benchmarks._common.search import _lower_is_better, _primary_metric
from benchmarks._common.sensitivity_report import (
    best_so_far,
    completed_values,
    incumbent_test_curve,
    load_study,
    render_plot,
    saturation_trial,
)
from benchmarks.datasets.download import default_dest
from benchmarks.datasets.registry import load

# Final-eval seed count per dataset used by the base run (match for comparable
# test metrics): small/medium 20, large single-holdout 10.
_FINAL_SEEDS = {"auto": 20, "heart": 20, "compas": 10, "blog": 10, "loan": 10}
_ORDER = ["heart", "auto", "compas", "blog", "loan"]


def _mode_residual(flavor: str) -> tuple[str, bool]:
    """Split a ``{mode}-{plain|residual}`` flavor label into ``(mode, residual)``.

    ``mixed-fixed-*`` maps to ``mode="mixed"`` (convex_fraction is a param, not a
    mode); its stored params omit ``convex_fraction``, so ``final_eval`` defaults
    it to 0.5 — exactly the fixed arm.

    :param flavor: Study flavor label, e.g. ``mixed-fixed-plain``.
    :returns: ``(mode, residual)``.
    """
    residual = "residual" in flavor or "deep" in flavor
    mode = "mixed" if flavor.startswith("mixed") else flavor.split("-")[0]
    return mode, residual


def extract_dataset(
    dataset: str,
    *,
    storage_dir: Path,
    test_seeds: int | None = None,
    with_test_curve: bool = True,
) -> dict[str, Any]:
    """Reconstruct one dataset's per-flavor sensitivity curves from storage.

    :param dataset: Dataset name; its study DBs are ``{storage_dir}/{dataset}-*.db``.
    :param storage_dir: Directory of committed Optuna sqlite DBs.
    :param test_seeds: Seeds per incumbent re-eval for Curve B; defaults to the
        base run's per-dataset count when ``None``.
    :param with_test_curve: When ``False``, skip Curve B (no `final_eval`).
    :returns: ``{"dataset", "flavors": {flavor: {"obj": [...], "test": [...]|None}},
        "rows": [saturation-row dicts]}``.
    """
    bundle = load(dataset, data_dir=default_dest())
    lower = _lower_is_better(_primary_metric(bundle))
    flavors: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for db in sorted(storage_dir.glob(f"{dataset}-*.db")):
        flavor = db.stem[len(dataset) + 1 :]
        study = load_study(db, db.stem)
        vals = completed_values(study, lower)
        if not vals:
            continue
        obj = best_so_far(vals, lower)
        t_star = saturation_trial(obj, lower)
        test_curve: list[float] | None = None
        n_reeval = 0
        if with_test_curve:
            mode, residual = _mode_residual(flavor)
            n_seeds = test_seeds or _FINAL_SEEDS.get(dataset, 10)
            test_curve, n_reeval = incumbent_test_curve(
                study,
                bundle,
                mode=mode,
                residual=residual,
                backend="torch",
                lower=lower,
                n_trials=len(vals),
                seeds=range(n_seeds),
            )
        flavors[flavor] = {"obj": obj, "test": test_curve}
        rows.append(
            {
                "dataset": dataset,
                "flavor": flavor,
                "trials": len(vals),
                "t_star": t_star,
                "saturated": t_star <= 0.8 * len(vals),
                "n_reeval": n_reeval,
            }
        )
    return {"dataset": dataset, "flavors": flavors, "rows": rows}


def saturation_table(rows: list[dict[str, Any]]) -> str:
    """Render the per-study saturation summary as a GitHub-flavored Markdown table.

    :param rows: One dict per study with keys ``dataset, flavor, trials, t_star,
        saturated, n_reeval``.
    :returns: The Markdown table as a single string.
    """
    out = [
        "| dataset | flavor | trials | t*(0.99) | saturated | # re-eval |",
        "|---|---|--:|--:|:-:|--:|",
    ]
    for r in rows:
        mark = "✅" if r["saturated"] else "⚠️"
        out.append(
            f"| {r['dataset']} | {r['flavor']} | {r['trials']} | "
            f"{r['t_star']} | {mark} | {r['n_reeval']} |"
        )
    return "\n".join(out)


def render_from_curves(curves: list[dict[str, Any]], out_path: Path) -> str:
    """Build the figure from per-dataset curve dicts and return the saturation table.

    :param curves: Per-dataset dicts as produced by :func:`extract_dataset`.
    :param out_path: Base path for the figure (png/pdf suffixes written).
    :returns: The Markdown saturation table across all datasets, in ``_ORDER``.
    """
    by_ds = {c["dataset"]: c for c in curves}
    series: dict[str, dict[str, tuple[list[float], list[float] | None]]] = {}
    rows: list[dict[str, Any]] = []
    for ds in _ORDER:
        c = by_ds.get(ds)
        if c is None:
            continue
        series[ds] = {fl: (d["obj"], d["test"]) for fl, d in c["flavors"].items()}
        rows.extend(c["rows"])
    render_plot(series, out_path)
    return saturation_table(rows)


def _cmd_extract(args: argparse.Namespace) -> None:
    curves = extract_dataset(
        args.dataset,
        storage_dir=Path(args.storage_dir),
        test_seeds=args.test_seeds,
        with_test_curve=not args.no_test_curve,
    )
    out = Path(args.curves_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.dataset}.json").write_text(json.dumps(curves))


def _cmd_render(args: argparse.Namespace) -> None:
    curves = [
        json.loads(p.read_text()) for p in sorted(Path(args.curves_dir).glob("*.json"))
    ]
    print(render_from_curves(curves, Path(args.out)))  # noqa: T201


def _cmd_run(args: argparse.Namespace) -> None:
    datasets = args.datasets or _ORDER
    curves_dir = Path(args.curves_dir)

    def _cmd(ds: str, device: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.sensitivity",
            "extract",
            "--dataset",
            ds,
            "--storage-dir",
            args.storage_dir,
            "--curves-dir",
            str(curves_dir),
        ]
        if args.test_seeds is not None:
            cmd += ["--test-seeds", str(args.test_seeds)]
        if args.no_test_curve:
            cmd += ["--no-test-curve"]
        return cmd

    fan_out(datasets, args.devices.split(","), _cmd, label=lambda d: f"extract={d}")
    curves = [
        json.loads((curves_dir / f"{ds}.json").read_text())
        for ds in datasets
        if (curves_dir / f"{ds}.json").exists()
    ]
    print(render_from_curves(curves, Path(args.out)))  # noqa: T201


def main() -> None:
    """Dispatch the ``extract`` / ``render`` / ``run`` subcommands."""
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="reconstruct one dataset's curves -> JSON")
    ex.add_argument("--dataset", required=True)
    ex.add_argument("--storage-dir", required=True)
    ex.add_argument("--curves-dir", required=True)
    ex.add_argument("--test-seeds", type=int, default=None)
    ex.add_argument("--no-test-curve", action="store_true")
    ex.set_defaults(func=_cmd_extract)

    rn = sub.add_parser("render", help="curve JSONs -> figure + saturation table")
    rn.add_argument("--curves-dir", required=True)
    rn.add_argument("--out", required=True)
    rn.set_defaults(func=_cmd_render)

    run = sub.add_parser("run", help="fan extract across GPUs, then render")
    run.add_argument("--storage-dir", required=True)
    run.add_argument("--curves-dir", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--datasets", nargs="*", default=None)
    run.add_argument("--devices", default="cuda:0,cuda:1")
    run.add_argument("--test-seeds", type=int, default=None)
    run.add_argument("--no-test-curve", action="store_true")
    run.set_defaults(func=_cmd_run)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
