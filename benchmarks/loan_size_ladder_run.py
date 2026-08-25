"""Size-ladder: whether deep monotone residual wins with scale (Stage B).

For each train size N and each arm (shallow D in [1,4] vs deep D in {6,10,16},
both absolute residual), tune HPs on an N-subsample, then refit + multi-seed
test on the full held-out test set (a fresh N-subsample per seed) and record the
IQM. Applies to any large dataset (`n_train >= 20_000`, see `_require_large`);
`--dataset` defaults to `loan` for back-compat. See
docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md.

Run: uv run --extra torch --group bench python -m benchmarks.loan_size_ladder_run \
    --dataset loan
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.results import ResultRow, interquartile_mean
from benchmarks._common.search import (
    _count_collapses,
    _lower_is_better,
    _primary_metric,
    _secondary_metrics,
    final_eval,
    search,
)
from benchmarks._common.splits import subsample_train

if TYPE_CHECKING:
    from collections.abc import Iterable

    from benchmarks._common.bundle import DatasetBundle

_NS: tuple[int, ...] = (5_000, 15_000, 45_000, 135_000, 1_000_000_000)  # last = full
_ARMS: tuple[str, ...] = ("shallow", "deep")

# Ladder-eligibility floor: a dataset needs at least this many training rows
# for the ladder's largest rungs (135_000, full) to probe scale meaningfully.
# Distinct from `search_spaces._LARGE_BATCH_THRESHOLD` (an unrelated batch-size
# band cutoff) even though both currently happen to be 20_000.
_MIN_LADDER_TRAIN = 20_000


def _require_large(bundle: DatasetBundle, dataset: str) -> None:
    """Raise if `bundle` is too small to size-ladder.

    Only large datasets (``n_train >= _MIN_LADDER_TRAIN``) can be
    meaningfully size-laddered: the ladder's largest rungs (135_000, full)
    are meant to probe scale, and a small dataset has no room to grow into
    them.

    :param bundle: The loaded dataset bundle.
    :param dataset: Dataset name, for the error message.
    :raises ValueError: If ``len(bundle.X_train) < _MIN_LADDER_TRAIN``.
    """
    n_train = len(bundle.X_train)
    if n_train < _MIN_LADDER_TRAIN:
        raise ValueError(
            f"size-ladder requires a large dataset (n_train >= "
            f"{_MIN_LADDER_TRAIN}); {dataset!r} has n_train={n_train}"
        )


def _ladder_eval(
    bundle: DatasetBundle,
    best_params: dict[str, Any],
    *,
    deep: bool,
    backend: str,
    n: int,
    final_seeds: Iterable[int],
    epochs: int,
) -> tuple[list[float], list[ResultRow]]:
    """Per-seed: subsample train to n (seed s), refit, test on full test.

    :returns: ``(values, rows)`` — the primary-metric value for each seed, and
        every underlying `ResultRow` (one per seed) concatenated across seeds,
        so a caller can also aggregate secondary metrics via
        `benchmarks._common.search._secondary_metrics`.
    """
    values: list[float] = []
    rows: list[ResultRow] = []
    for s in final_seeds:
        b_s = subsample_train(bundle, n, seed=s)
        agg, seed_rows = final_eval(
            b_s,
            best_params,
            mode="absolute",
            residual=True,
            backend=backend,
            seeds=[s],
            epochs=epochs,
        )
        values.append(float(agg.values[0]))
        rows.extend(seed_rows)
    return values, rows


def run_ladder(
    bundle: DatasetBundle,
    *,
    dataset: str | None = None,
    ns: tuple[int, ...] = _NS,
    arms: tuple[str, ...] = _ARMS,
    backend: str = "torch",
    n_trials: int = 25,
    search_seeds: int = 3,
    final_seeds: Iterable[int] = range(10),
    epochs: int = 50,
    n_splits: int = 1,
    n_jobs: int = 1,
) -> list[dict[str, Any]]:
    """Run the size-ladder for `bundle`; return one record per (n, arm).

    Tunes/reports on the dataset's primary metric (``roc_auc`` for
    classification, ``mse`` for regression; see `_primary_metric`), not a
    hardcoded metric — so this generalizes beyond `loan`-style datasets.

    When `dataset` is not None (every real Stage-B / CLI call passes it), the
    large-dataset guard (`_require_large`, `n_train >= 20_000`) is enforced at
    the function boundary. Internal callers that pass `dataset=None` (e.g. the
    single-point `screen_dataset` in `large_screen_run.py`, or the smoke test)
    stay unguarded — they intentionally run tiny/arbitrary-size bundles.

    :param dataset: Dataset name to stamp on each record and gate on; defaults
        to `bundle.name` for the record stamp. Passing a non-None value also
        triggers the `n_train >= 20_000` eligibility check. Does not affect
        what is loaded (`bundle` is already loaded).
    :param n_jobs: Optuna trial parallelism within each arm's search (threaded).
    :raises ValueError: If `dataset is not None` and the bundle is too small
        to size-ladder (`n_train < 20_000`).
    """
    if dataset is not None:
        _require_large(bundle, dataset)
    ds_name = dataset if dataset is not None else bundle.name
    metric = _primary_metric(bundle)
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
                n_jobs=n_jobs,
            )
            values, eval_rows = _ladder_eval(
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
                    "dataset": ds_name,
                    "n": n_eff,
                    "arm": arm,
                    "depth": int(study.best_params["depth"]),
                    "best_params": study.best_params,
                    "cv_best": study.best_value,
                    "test_metric": metric,
                    "test_mean": float(np.mean(values)),
                    "test_std": float(np.std(values)),
                    "test_median": float(np.median(values)),
                    "test_iqm": interquartile_mean(np.asarray(values)),
                    "test_values": values,
                    # Secondary metrics (e.g. accuracy alongside roc_auc);
                    # empty for regression datasets (single metric).
                    "secondary": _secondary_metrics(eval_rows, metric),
                    "n_collapse": _count_collapses(
                        tuple(values),
                        task=bundle.task,
                        base_rate=base_rate,
                        lower_is_better=_lower_is_better(metric),
                        metric=metric,
                    ),
                    "n_seeds": len(values),
                }
            )
    return records


_DEFAULT_DATASET = "loan"


def _default_out(dataset: str) -> Path:
    """Canonical results path for `dataset`'s size-ladder run."""
    return (
        Path(__file__).resolve().parent / "results" / "size-ladder" / f"{dataset}.json"
    )


def main() -> None:
    """CLI: run a (subset of the) ladder and write records JSON.

    With no arguments, runs the full ladder for `loan` (back-compat default)
    to the canonical results path. ``--dataset`` selects any other large
    dataset from the registry (`n_train >= _MIN_LADDER_TRAIN`; smaller
    datasets raise `ValueError` via `_require_large`). The
    ``--ns``/``--arms``/``--out`` options let a launcher run one cell per
    process (each pinned to a GPU via ``$MONONET_TORCH_DEVICE``) into a partial
    file, to be merged afterwards.
    """
    import argparse

    ap = argparse.ArgumentParser(description="size-ladder run for a large dataset")
    ap.add_argument("--dataset", default=_DEFAULT_DATASET, help="dataset key")
    ap.add_argument(
        "--ns",
        default=",".join(str(n) for n in _NS),
        help="comma-separated train sizes (use 1000000000 for the full split)",
    )
    ap.add_argument("--arms", default=",".join(_ARMS), help="comma-separated arms")
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    out: Path = args.out if args.out is not None else _default_out(args.dataset)

    bundle = load(args.dataset, data_dir=default_dest())
    # The n_train >= 20_000 guard fires inside run_ladder (dataset is not None).
    records = run_ladder(
        bundle,
        dataset=args.dataset,
        ns=tuple(int(x) for x in args.ns.split(",")),
        arms=tuple(args.arms.split(",")),
        n_trials=args.n_trials,
        search_seeds=args.search_seeds,
        final_seeds=range(args.final_seeds),
        epochs=args.epochs,
        n_jobs=args.n_jobs,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {out} ({len(records)} records)")  # noqa: T201


if __name__ == "__main__":
    main()
