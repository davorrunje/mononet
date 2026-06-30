"""Optuna search engine over the Phase-1 run() harness (validation-driven)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.results import Aggregate, aggregate
from benchmarks._common.runner import run
from benchmarks._common.search_spaces import suggest_config
from benchmarks._common.splits import cv_splits

if TYPE_CHECKING:
    from collections.abc import Iterable

    from benchmarks._common.bundle import DatasetBundle


def flavor_name(mode: str, residual: bool) -> str:
    return f"{mode}-{'residual' if residual else 'plain'}"


def _primary_metric(bundle: DatasetBundle) -> str:
    return "accuracy" if bundle.task == "binary_classification" else "mse"


def _lower_is_better(metric: str) -> bool:
    return metric in ("mse", "rmse")


@dataclass(frozen=True, slots=True)
class StudyResult:
    dataset: str
    flavor: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int


def _fold_bundles(
    bundle: DatasetBundle, *, n_splits: int, seed: int
) -> list[DatasetBundle]:
    """Throwaway per-fold bundles with each fold's validation rows in the test slot.

    Lets the search reuse run() (which evaluates on X_test) to score on every CV
    fold without ever touching the real test set.
    """
    folds = cv_splits(bundle, n_splits=n_splits, seed=seed)
    out: list[DatasetBundle] = []
    for tr, val in folds:
        out.append(
            dataclasses.replace(
                bundle,
                X_train=bundle.X_train[tr],
                y_train=bundle.y_train[tr],
                X_test=bundle.X_train[val],
                y_test=bundle.y_train[val],
            )
        )
    return out


def search(
    bundle: DatasetBundle,
    *,
    mode: str,
    residual: bool,
    backend: str,
    n_trials: int = 50,
    seed: int = 0,
    epochs: int = 50,
    n_jobs: int = 1,
    n_splits: int = 5,
    metric: str | None = None,
    storage: str | None = None,
) -> StudyResult:
    """Tune (dataset, flavor) HPs by mean k-fold CV metric via Optuna TPE."""
    metric = metric or _primary_metric(bundle)
    direction = "minimize" if _lower_is_better(metric) else "maximize"
    folds = _fold_bundles(bundle, n_splits=n_splits, seed=seed)

    def objective(trial: optuna.Trial) -> float:
        cfg: BenchmarkConfig = suggest_config(
            trial,
            dataset=bundle.name,
            backend=backend,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            residual=residual,
            epochs=epochs,  # type: ignore[arg-type]
            metric=metric,  # type: ignore[arg-type]
        )
        scores: list[float] = []
        for fb in folds:
            rows = run(cfg, fb)
            if not rows:
                raise RuntimeError("run() returned no rows for trial")
            scores.append(float(rows[0].scores[metric]))  # type: ignore[index]
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=f"{bundle.name}-{flavor_name(mode, residual)}",
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return StudyResult(
        dataset=bundle.name,
        flavor=flavor_name(mode, residual),
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
    )


def final_eval(
    bundle: DatasetBundle,
    best_params: dict[str, Any],
    *,
    mode: str,
    residual: bool,
    backend: str,
    metric: str | None = None,
    seeds: Iterable[int] = range(10),
    epochs: int = 50,
) -> Aggregate:
    """Refit best HPs on the full train split; report TEST mean±std over all seeds."""
    metric = metric or _primary_metric(bundle)
    width = int(best_params["width"])
    cfg = BenchmarkConfig(
        dataset=bundle.name,
        backend=backend,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        residual=residual,  # type: ignore[arg-type]
        depth=int(best_params["depth"]),
        width=width,
        activation="elu",
        convex_fraction=float(best_params.get("convex_fraction", 0.5)),
        embed_hidden=(width,),
        dropout=float(best_params["dropout"]),
        optimizer=OptimizerSpec(
            "adam", float(best_params["lr"]), float(best_params["weight_decay"])
        ),
        lr_decay=float(best_params["lr_decay"]),
        batch_size=int(best_params["batch_size"]),
        epochs=epochs,
        early_stopping=None,
        seeds=tuple(seeds),
        metrics=(metric,),  # type: ignore[arg-type]
    )
    rows = run(cfg, bundle)
    return aggregate(
        rows, metric=metric, lower_is_better=_lower_is_better(metric), top_k=len(rows)
    )


_ALL_FLAVORS: tuple[tuple[str, bool], ...] = (
    ("switch", False),
    ("switch", True),
    ("absolute", False),
    ("absolute", True),
)
# (n_trials, final_seeds, n_splits) per dataset.
# n_splits: 5-fold CV for small/medium datasets; 1 (single holdout) for the large
# ones (loan/blog), where a single split is already low-variance and 5x cheaper.
_BUDGET: dict[str, tuple[int, range, int]] = {
    "auto": (50, range(10), 5),
    "heart": (50, range(10), 5),
    "compas": (50, range(10), 5),
    "loan": (25, range(5), 1),
    "blog": (25, range(5), 1),
}


def run_dataset(
    dataset: str,
    *,
    backend: str = "torch",
    flavors: tuple[tuple[str, bool], ...] = _ALL_FLAVORS,
    n_trials: int | None = None,
    epochs: int = 50,
    n_jobs: int = 1,
    final_seeds: Iterable[int] | None = None,
    n_splits: int | None = None,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    storage_dir: Path | None = None,
) -> list[Path]:
    """Search + final_eval each flavor of one dataset; write per-flavor JSON.

    Budget falls back to the per-dataset `_BUDGET` defaults when not overridden.
    Returns the written JSON paths.
    """
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    b_trials, b_seeds, b_splits = _BUDGET.get(dataset, (50, range(10), 5))
    n_trials = b_trials if n_trials is None else n_trials
    final_seeds = b_seeds if final_seeds is None else final_seeds
    n_splits = b_splits if n_splits is None else n_splits
    data_dir = data_dir or default_dest()
    out_dir = out_dir or (Path(__file__).resolve().parents[1] / "results" / "phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load(dataset, data_dir=data_dir)
    written: list[Path] = []
    for mode, residual in flavors:
        fname = flavor_name(mode, residual)
        storage = (
            f"sqlite:///{storage_dir}/{dataset}-{fname}.db" if storage_dir else None
        )
        study = search(
            bundle,
            mode=mode,
            residual=residual,
            backend=backend,
            n_trials=n_trials,
            epochs=epochs,
            n_jobs=n_jobs,
            n_splits=n_splits,
            storage=storage,
        )
        agg = final_eval(
            bundle,
            study.best_params,
            mode=mode,
            residual=residual,
            backend=backend,
            seeds=final_seeds,
            epochs=epochs,
        )
        rec = {
            "dataset": dataset,
            "flavor": study.flavor,
            "best_params": study.best_params,
            "cv_best": study.best_value,
            "test_metric": agg.metric,
            "test_mean": agg.mean,
            "test_std": agg.std,
            "n_seeds": agg.n_seeds,
        }
        path = out_dir / f"{dataset}-{fname}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        written.append(path)
    return written
