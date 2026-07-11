"""Within-loan size-ladder: whether deep monotone residual wins with scale.

For each train size N and each arm (shallow D in [1,4] vs deep D in {6,10,16},
both absolute residual), tune HPs on an N-subsample, then refit + multi-seed
test on the full held-out test set (a fresh N-subsample per seed) and record the
IQM. See docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md.

Run: uv run --extra torch --group bench python -m benchmarks.loan_size_ladder_run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.results import interquartile_mean
from benchmarks._common.search import _count_collapses, final_eval, search
from benchmarks._common.splits import subsample_train

if TYPE_CHECKING:
    from collections.abc import Iterable

    from benchmarks._common.bundle import DatasetBundle

_NS: tuple[int, ...] = (5_000, 15_000, 45_000, 135_000, 1_000_000_000)  # last = full
_ARMS: tuple[str, ...] = ("shallow", "deep")


def _ladder_eval(
    bundle: DatasetBundle,
    best_params: dict[str, Any],
    *,
    deep: bool,
    backend: str,
    n: int,
    final_seeds: Iterable[int],
    epochs: int,
) -> list[float]:
    """Per-seed: subsample train to n (seed s), refit, test on full test."""
    values: list[float] = []
    for s in final_seeds:
        b_s = subsample_train(bundle, n, seed=s)
        agg = final_eval(
            b_s,
            best_params,
            mode="absolute",
            residual=True,
            backend=backend,
            seeds=[s],
            epochs=epochs,
        )
        values.append(float(agg.values[0]))
    return values


def run_ladder(
    bundle: DatasetBundle,
    *,
    ns: tuple[int, ...] = _NS,
    arms: tuple[str, ...] = _ARMS,
    backend: str = "torch",
    n_trials: int = 25,
    search_seeds: int = 3,
    final_seeds: Iterable[int] = range(10),
    epochs: int = 50,
    n_splits: int = 1,
) -> list[dict[str, Any]]:
    """Run the size-ladder for `bundle`; return one record per (n, arm)."""
    seeds = list(final_seeds)
    base_rate = max(float(np.mean(bundle.y_test)), 1.0 - float(np.mean(bundle.y_test)))
    records: list[dict[str, Any]] = []
    for n in ns:
        for arm in arms:
            deep = arm == "deep"
            b_search = subsample_train(bundle, n, seed=0)
            study = search(
                b_search,
                mode="absolute",
                residual=True,
                deep=deep,
                backend=backend,
                n_trials=n_trials,
                epochs=epochs,
                n_splits=n_splits,
                search_seeds=search_seeds,
            )
            values = _ladder_eval(
                bundle,
                study.best_params,
                deep=deep,
                backend=backend,
                n=n,
                final_seeds=seeds,
                epochs=epochs,
            )
            n_eff = min(n, len(bundle.X_train))
            records.append(
                {
                    "n": n_eff,
                    "arm": arm,
                    "depth": int(study.best_params["depth"]),
                    "best_params": study.best_params,
                    "cv_best": study.best_value,
                    "test_metric": "accuracy",
                    "test_mean": float(np.mean(values)),
                    "test_std": float(np.std(values)),
                    "test_median": float(np.median(values)),
                    "test_iqm": interquartile_mean(np.asarray(values)),
                    "test_values": values,
                    "n_collapse": _count_collapses(
                        tuple(values),
                        task=bundle.task,
                        base_rate=base_rate,
                        lower_is_better=False,
                    ),
                    "n_seeds": len(values),
                }
            )
    return records


def main() -> None:
    """Load loan, run the full ladder, write the committed results JSON."""
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    bundle = load("loan", data_dir=default_dest())
    records = run_ladder(bundle)
    out = Path(__file__).resolve().parent / "results" / "size-ladder" / "loan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {out} ({len(records)} records)")  # noqa: T201


if __name__ == "__main__":
    main()
