"""Max-size deep/shallow screen: one Δ + gate verdict per dataset.

Runs the standard search for the deep and shallow ``mixed``-residual arms at
the dataset's full train size, multi-seed refit + test, and gates on
Δ = IQM(deep) - IQM(shallow). See
docs/superpowers/specs/2026-07-11-large-dataset-screen-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks._common.screen_gate import DEFAULT_MARGIN, gate
from benchmarks._common.size_ladder_report import delta_by_n
from benchmarks.loan_size_ladder_run import run_ladder

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle

_FULL = 1_000_000_000  # >= any train size ⇒ subsample_train returns the full split


def screen_dataset(
    bundle: DatasetBundle,
    *,
    n_trials: int = 25,
    search_seeds: int = 3,
    final_seeds: int = 10,
    epochs: int = 50,
    n_jobs: int = 1,
    backend: str = "torch",
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Screen one dataset at full size; return the record + gate verdict."""
    recs = run_ladder(
        bundle,
        ns=(_FULL,),
        arms=("deep", "shallow"),
        backend=backend,
        n_trials=n_trials,
        search_seeds=search_seeds,
        final_seeds=range(final_seeds),
        epochs=epochs,
        n_jobs=n_jobs,
    )
    d = delta_by_n(recs)[0]
    return {
        "name": bundle.name,
        "n_full": d["n"],
        "deep_iqm": d["deep_iqm"],
        "shallow_iqm": d["shallow_iqm"],
        "delta": d["delta"],
        "delta_lo": d["delta_lo"],
        "delta_hi": d["delta_hi"],
        "margin": margin,
        "verdict": gate(d["delta_lo"], d["delta"], margin),
    }


def main() -> None:
    """CLI: screen one dataset and write its record JSON."""
    import argparse

    from benchmarks.datasets.registry import load
    from benchmarks.datasets.sources import require_dataset

    ap = argparse.ArgumentParser(description="max-size deep/shallow screen")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bundle = load(args.dataset, data_dir=require_dataset(args.dataset))
    rec = screen_dataset(
        bundle,
        n_trials=args.n_trials,
        search_seeds=args.search_seeds,
        final_seeds=args.final_seeds,
        epochs=args.epochs,
        n_jobs=args.n_jobs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"{args.dataset}: Δ={rec['delta']:+.4f} verdict={rec['verdict']}")  # noqa: T201


if __name__ == "__main__":
    main()
