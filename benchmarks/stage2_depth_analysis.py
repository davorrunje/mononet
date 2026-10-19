"""Stage-2 depth analysis recast around the L=4 sufficiency threshold.

By the depth-4 monotone universal-approximation theorem (Mikulincer-Reichman;
formalized in the `neural-network-proofs` repo), **4 monotone layers already
approximate any monotone function**, so ``L = 4`` is the *sufficiency threshold*
and anything beyond it is *excess depth*. This module recasts the Stage-2
results on that axis rather than the plain/residual/deep flavor split:

* effective monotone layers ``L``: ``plain`` -> ``depth + 1``; ``residual`` /
  ``deep`` -> ``2*depth + 2`` (an input projection + ``depth`` blocks of
  ``sub_depth=2`` + a head);
* per dataset, the best model at ``L <= 4`` (sufficient) vs ``L > 4`` (excess),
  in the dataset's primary metric (MSE lower-better; ROC-AUC higher-better);
* an error-vs-L curve from the Optuna search trials (the best CV objective
  achieved at each sampled ``L``), with the ``L = 4`` line marked.

Reads committed per-flavor result JSON in ``benchmarks/results/stage2`` and the
resumable study DBs in ``.../studies`` (trials, for the curve). No training here.

Run: ``uv run --extra torch --group bench python -m benchmarks.stage2_depth_analysis``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_SUFFICIENT_L = 4
_LOWER_BETTER = frozenset({"mse", "rmse"})


def effective_layers(flavor: str, depth: int) -> int:
    """Effective monotone-layer count ``L`` for a flavor at a given ``depth``.

    :param flavor: Flavor label like ``"absolute-residual"`` / ``"switch-deep"``.
    :param depth: The ``depth`` hyperparameter (block count).
    :returns: ``depth + 1`` for ``plain`` flavors, else ``2*depth + 2``.
    """
    return depth + 1 if flavor.endswith("plain") else 2 * depth + 2


def _final_points(results_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Per-dataset list of tuned per-flavor points from the result JSON."""
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for f in results_dir.glob("*.json"):
        d = json.loads(f.read_text())
        flavor = str(d["flavor"])
        depth = int(d["best_params"]["depth"])
        per[str(d["dataset"])].append(
            {
                "flavor": flavor,
                "depth": depth,
                "L": effective_layers(flavor, depth),
                "metric": str(d["test_metric"]),
                "value": float(d["test_iqm"]),
            }
        )
    return per


def recast_rows(results_dir: Path) -> list[dict[str, Any]]:
    """Best-at-``L<=4`` vs best-at-``L>4`` per *complete* (6-flavor) dataset.

    :param results_dir: Directory of per-flavor Stage-2 result JSON files.
    :returns: One dict per complete dataset with the sufficiency-vs-excess split.
    """
    out: list[dict[str, Any]] = []
    for ds, pts in sorted(_final_points(results_dir).items()):
        if len(pts) < 6:
            continue
        metric = pts[0]["metric"]
        lower = metric in _LOWER_BETTER
        pick = min if lower else max
        suff = [p for p in pts if p["L"] <= _SUFFICIENT_L]
        exc = [p for p in pts if p["L"] > _SUFFICIENT_L]
        best_suff = pick(suff, key=lambda p: p["value"]) if suff else None
        best_exc = pick(exc, key=lambda p: p["value"]) if exc else None
        delta: float | None = None
        if best_suff and best_exc:
            delta = (
                (best_suff["value"] - best_exc["value"])
                if lower
                else (best_exc["value"] - best_suff["value"])
            )
        out.append(
            {
                "dataset": ds,
                "metric": metric,
                "points": sorted(pts, key=lambda p: p["L"]),
                "best_suff": best_suff,
                "best_exc": best_exc,
                "delta_excess_minus_suff": delta,
            }
        )
    return out


def error_vs_layers(studies_dir: Path, dataset: str, metric: str) -> dict[int, float]:
    """Best CV objective achieved at each sampled ``L`` (from the search trials).

    :param studies_dir: Directory of Optuna ``*.db`` study files.
    :param dataset: Dataset key.
    :param metric: Primary metric (sets the better-direction).
    :returns: Map ``L -> best CV objective`` over all trials of all flavors.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    lower = metric in _LOWER_BETTER
    by_l: dict[int, list[float]] = defaultdict(list)
    for db in studies_dir.glob(f"{dataset}-*.db"):
        flavor = db.name[len(dataset) + 1 : -3]
        storage = f"sqlite:///{db}"
        for summ in optuna.study.get_all_study_summaries(storage=storage):
            s = optuna.load_study(study_name=summ.study_name, storage=storage)
            for t in s.trials:
                depth = t.params.get("depth")
                if (
                    t.state.name == "COMPLETE"
                    and t.value is not None
                    and depth is not None
                ):
                    by_l[effective_layers(flavor, int(depth))].append(float(t.value))
    return {lvl: (min(v) if lower else max(v)) for lvl, v in by_l.items()}


def _fmt(p: dict[str, Any] | None) -> str:
    return "-" if p is None else f"{p['flavor']}(L={p['L']}) {p['value']:.4f}"


def print_table(results_dir: Path) -> None:
    """Print the sufficiency-vs-excess recast for all complete datasets."""
    rows = recast_rows(results_dir)
    if not rows:
        print("(no complete datasets yet)")  # noqa: T201
        return
    print(  # noqa: T201
        f"{'dataset':26} {'metric':7} {'best L<=4 (sufficient)':30} "
        f"{'best L>4 (excess)':30} {'excess helps?':14}"
    )
    for r in rows:
        d = r["delta_excess_minus_suff"]
        if d is None:
            verdict = "n/a (bin empty)"
        elif d > 0:
            verdict = f"yes +{d:.4f}"
        else:
            verdict = f"NO {d:+.4f}"
        print(  # noqa: T201
            f"{r['dataset']:26} {r['metric']:7} {_fmt(r['best_suff']):30} "
            f"{_fmt(r['best_exc']):30} {verdict:14}"
        )


def render_plot(results_dir: Path, studies_dir: Path, out: Path) -> None:
    """Save an error-vs-L facet plot (one panel per complete dataset)."""
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = recast_rows(results_dir)
    if not rows:
        print("(no complete datasets yet; no plot)")  # noqa: T201
        return
    n = len(rows)
    cols = min(3, n)
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(
        nrows, cols, figsize=(5 * cols, 3.2 * nrows), squeeze=False
    )
    for i, row in enumerate(rows):
        ax = axes[i // cols][i % cols]
        curve = error_vs_layers(studies_dir, row["dataset"], row["metric"])
        if curve:
            xs = sorted(curve)
            ax.plot(xs, [curve[x] for x in xs], "o-", ms=4, label="best CV vs L")
        for p in row["points"]:
            ax.plot(p["L"], p["value"], "x", color="crimson")
        ax.axvline(_SUFFICIENT_L, ls="--", color="gray", lw=1)
        ax.set_title(f"{row['dataset']} ({row['metric']})", fontsize=9)
        ax.set_xlabel("effective layers L")
        ax.set_ylabel(row["metric"])
    for j in range(n, nrows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(
        "Error vs effective depth L (dashed = L=4 sufficiency; red mark = test)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")  # noqa: T201


def main() -> None:
    """CLI: print the recast table and optionally render the error-vs-L plot."""
    root = Path(__file__).resolve().parent / "results" / "stage2"
    ap = argparse.ArgumentParser(description="Stage-2 depth recast (L=4 threshold)")
    ap.add_argument("--results-dir", type=Path, default=root)
    ap.add_argument("--studies-dir", type=Path, default=root / "studies")
    ap.add_argument("--plot", type=Path, default=None, help="save error-vs-L plot here")
    args = ap.parse_args()
    print_table(args.results_dir)
    if args.plot is not None:
        render_plot(args.results_dir, args.studies_dir, args.plot)


if __name__ == "__main__":
    main()
