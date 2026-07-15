"""CLI: reconstruct HP-search sensitivity curves from committed Optuna storage.

Reconstructs Curve A (best-so-far objective) and, unless ``--no-test-curve``,
Curve B (test-of-incumbent, bounded re-eval) for each ``(dataset, flavor)``
study under ``--storage-dir``; writes the faceted figure to ``--out`` and prints
a Markdown saturation table. Reads storage only — it never re-runs the search::

    uv run --group bench python -m benchmarks.sensitivity \
        --storage-dir benchmarks/results/alternate-base/studies \
        --out docs/_static/hp-search-sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

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


def main() -> None:
    """Reconstruct curves for each study under ``--storage-dir`` and emit outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=_ORDER)
    ap.add_argument("--flavors", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-test-curve", action="store_true")
    ap.add_argument(
        "--test-seeds",
        type=int,
        default=None,
        help=(
            "Seeds per incumbent re-eval for Curve B (trend line). Defaults to "
            "the base run's per-dataset counts; use a smaller value (e.g. 5) for "
            "a cheaper trend estimate."
        ),
    )
    args = ap.parse_args()

    store = Path(args.storage_dir)
    series: dict[str, dict[str, tuple[list[float], list[float] | None]]] = {}
    table_rows: list[dict[str, Any]] = []
    for ds in args.datasets:
        bundle = load(ds, data_dir=default_dest())
        metric = _primary_metric(bundle)
        lower = _lower_is_better(metric)
        series[ds] = {}
        for db in sorted(store.glob(f"{ds}-*.db")):
            flavor = db.stem[len(ds) + 1 :]
            if args.flavors and flavor not in args.flavors:
                continue
            study = load_study(db, db.stem)
            vals = completed_values(study, lower)
            if not vals:
                continue
            obj = best_so_far(vals, lower)
            t_star = saturation_trial(obj, lower)
            test_curve: list[float] | None = None
            n_reeval = 0
            if not args.no_test_curve:
                mode, residual = _mode_residual(flavor)
                n_seeds = args.test_seeds or _FINAL_SEEDS.get(ds, 10)
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
            series[ds][flavor] = (obj, test_curve)
            table_rows.append(
                {
                    "dataset": ds,
                    "flavor": flavor,
                    "trials": len(vals),
                    "t_star": t_star,
                    "saturated": t_star <= 0.8 * len(vals),
                    "n_reeval": n_reeval,
                }
            )
    render_plot(series, Path(args.out))
    print(saturation_table(table_rows))  # noqa: T201


if __name__ == "__main__":
    main()
