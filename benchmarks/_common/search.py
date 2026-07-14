"""Optuna search engine over the Phase-1 run() harness (cross-validation-driven)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.results import Aggregate, ResultRow, aggregate
from benchmarks._common.runner import run
from benchmarks._common.search_spaces import suggest_config
from benchmarks._common.splits import cv_splits

if TYPE_CHECKING:
    from collections.abc import Iterable

    from benchmarks._common.bundle import DatasetBundle


def flavor_name(mode: str, residual: bool, deep: bool = False) -> str:
    """Canonical flavor label for result files and Optuna study names.

    :param mode: Monotonicity mode (``"split"`` or ``"mixed"``).
    :param residual: Whether the stack uses residual blocks.
    :param deep: Whether this is the deep-depth-band flavor. When ``True`` the
        label is ``"{mode}-deep"`` regardless of ``residual`` (deep implies
        residual); otherwise ``"{mode}-residual"`` or ``"{mode}-plain"``.
    :returns: The flavor label string.
    """
    if deep:
        return f"{mode}-deep"
    return f"{mode}-{'residual' if residual else 'plain'}"


def _primary_metric(bundle: DatasetBundle) -> str:
    return "roc_auc" if bundle.task == "binary_classification" else "mse"


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
    deep: bool = False,
    n_trials: int = 50,
    seed: int = 0,
    epochs: int = 50,
    n_jobs: int = 1,
    n_splits: int = 5,
    search_seeds: int = 3,
    metric: str | None = None,
    storage: str | None = None,
    search_activation: bool = False,
    max_depth: int = 4,
    embed_layers: int = 1,
) -> StudyResult:
    """Tune (dataset, flavor) HPs by a **stability-aware** k-fold CV objective.

    Each trial is scored over ``n_splits`` folds x ``search_seeds`` seeds, and
    the objective is the risk-adjusted (one-sigma) bound rather than the plain
    mean: ``mean - std`` for maximize metrics, ``mean + std`` for minimize
    metrics. Evaluating several seeds per fold exposes seed-dependent training
    collapses (which a single-seed CV misses), and the variance penalty steers
    the search away from fragile HP regions that train well on average but
    collapse on some seeds. See [[stage2-collapse-investigation]].

    :param search_activation: When ``True``, sample ``activation`` from
        ``{"relu", "elu", "softplus", "selu"}``; otherwise fix it to ``"elu"``.
    :param max_depth: Upper bound of the shallow ``depth`` range (``[1,
        max_depth]``) used when ``deep`` is `False`.
    :param embed_layers: Number of non-monotone `Dense` layers in
        ``cfg.embed_hidden``, each sized ``width``.
    """
    metric = metric or _primary_metric(bundle)
    lower = _lower_is_better(metric)
    direction = "minimize" if lower else "maximize"
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
            n_train=int(bundle.X_train.shape[0]),
            deep=deep,
            search_activation=search_activation,
            max_depth=max_depth,
            embed_layers=embed_layers,
        )
        cfg = dataclasses.replace(cfg, seeds=tuple(range(search_seeds)))
        scores: list[float] = []
        for fb in folds:
            rows = run(cfg, fb)
            if not rows:
                raise RuntimeError("run() returned no rows for trial")
            scores.extend(float(r.scores[metric]) for r in rows)  # type: ignore[index]
        arr = np.asarray(scores, dtype=np.float64)
        # Risk-adjusted objective: penalise seed variance so unstable HP regions
        # (good mean, occasional collapse) are not selected.
        return float(arr.mean() + arr.std()) if lower else float(arr.mean() - arr.std())

    study = optuna.create_study(
        study_name=f"{bundle.name}-{flavor_name(mode, residual, deep)}",
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return StudyResult(
        dataset=bundle.name,
        flavor=flavor_name(mode, residual, deep),
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
    embed_layers: int = 1,
) -> tuple[Aggregate, list[ResultRow]]:
    """Refit best HPs on the full train split; report TEST mean±std over all seeds.

    :param embed_layers: Number of non-monotone `Dense` layers in
        ``cfg.embed_hidden``, each sized ``width``.
    :returns: ``(agg, rows)`` — the primary-metric :class:`Aggregate` and the
        raw per-seed :class:`ResultRow` list backing it. Each row's ``scores``
        holds every metric computed for that seed (e.g. ``roc_auc`` *and*
        ``accuracy`` for classification, per the `metrics` tuple below), so
        callers can aggregate secondary metrics (see `_secondary_metrics`)
        without a second training run.
    """
    metric = metric or _primary_metric(bundle)
    metrics: tuple[str, ...] = (
        ("roc_auc", "accuracy") if metric == "roc_auc" else (metric,)
    )
    width = int(best_params["width"])
    cfg = BenchmarkConfig(
        dataset=bundle.name,
        backend=backend,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        residual=residual,  # type: ignore[arg-type]
        depth=int(best_params["depth"]),
        width=width,
        activation=str(best_params.get("activation", "elu")),  # type: ignore[arg-type]
        convex_fraction=float(best_params.get("convex_fraction", 0.5)),
        embed_hidden=tuple(width for _ in range(embed_layers)),
        dropout=float(best_params["dropout"]),
        optimizer=OptimizerSpec(
            "adam", float(best_params["lr"]), float(best_params["weight_decay"])
        ),
        lr_decay=float(best_params["lr_decay"]),
        batch_size=int(best_params["batch_size"]),
        epochs=epochs,
        early_stopping=None,
        seeds=tuple(seeds),
        metrics=metrics,  # type: ignore[arg-type]
        alt_init="composition" if mode == "alternate" else None,
    )
    rows = run(cfg, bundle)
    agg = aggregate(
        rows, metric=metric, lower_is_better=_lower_is_better(metric), top_k=len(rows)
    )
    return agg, rows


def _secondary_metrics(
    rows: list[ResultRow], primary_metric: str
) -> dict[str, dict[str, Any]]:
    """Aggregate every metric in ``rows[0].scores`` other than ``primary_metric``.

    Shared by `run_dataset` and the size-ladder's `run_ladder` so both persist
    secondary metrics (e.g. classification ``accuracy`` alongside the primary
    ``roc_auc``) the same way, reusing :func:`aggregate` per metric.

    :param rows: Per-seed result rows (from `final_eval`, possibly
        concatenated across several `final_eval` calls, e.g. one per
        size-ladder seed); each row's ``scores`` holds every metric computed
        for that run.
    :param primary_metric: The metric already reported under ``test_*``;
        excluded here to avoid duplicating it.
    :returns: ``{metric: {"iqm", "mean", "std", "median", "values"}}`` for
        every other metric present in ``rows``; empty when there is none
        (e.g. regression, which has a single metric).
    """
    if not rows:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for m in rows[0].scores:
        if m == primary_metric:
            continue
        agg = aggregate(
            rows, metric=m, lower_is_better=_lower_is_better(m), top_k=len(rows)
        )
        out[m] = {
            "iqm": agg.iqm,
            "mean": agg.mean,
            "std": agg.std,
            "median": agg.median,
            "values": list(agg.values),
        }
    return out


# (mode, residual, deep) triples. Deep implies residual=True with a larger
# depth search band (see suggest_config); it is a separate Optuna study.
_ALL_FLAVORS: tuple[tuple[str, bool, bool], ...] = (
    ("split", False, False),
    ("split", True, False),
    ("mixed", False, False),
    ("mixed", True, False),
    ("split", True, True),
    ("mixed", True, True),
)
# (n_trials, final_seeds, n_splits) per dataset.
# n_splits: 5-fold CV for small/medium datasets; 1 (single holdout) for the large
# ones (compas/loan/blog), where a single split is already low-variance and 5x
# cheaper. final_seeds bumped to 20 for the small/medium datasets so the robust
# estimators (median, IQM) and the collapse count are stable; compas/loan/blog
# keep a smaller count (their single-holdout final_eval is already
# near-deterministic, std ~1e-4).
# n_trials for auto/heart/compas/blog/loan match the paper's (airtai/
# monotonic-nn) per-dataset Optuna trial counts: AutoMPG/heart=200;
# compas/blog/loan=50. compas also switches to a single holdout split (1),
# like the other larger datasets.
_BUDGET: dict[str, tuple[int, range, int]] = {
    "auto": (200, range(20), 5),
    "heart": (200, range(20), 5),
    "compas": (50, range(10), 1),
    "loan": (50, range(10), 1),
    "blog": (50, range(10), 1),
    "adult": (25, range(10), 5),
    "taiwan": (25, range(10), 5),
    "polish": (25, range(10), 5),
    "german": (25, range(10), 5),
    "lc": (25, range(10), 1),
}
# Cheap budget for every generator-backed `synth_*` depth-probe dataset
# (#99): small trial/seed counts since generation (and eval) is fast and
# these are ablations, not headline results.
_SYNTH_BUDGET: tuple[int, range, int] = (25, range(5), 1)


def _budget_for(dataset: str) -> tuple[int, range, int]:
    """Resolve the ``(n_trials, final_seeds, n_splits)`` budget for *dataset*.

    :param dataset: Dataset key.
    :returns: Explicit :data:`_BUDGET` entry; :data:`_SYNTH_BUDGET` for any
        ``synth_*`` key; else the global default.
    """
    if dataset.startswith("synth_"):
        return _SYNTH_BUDGET
    return _BUDGET.get(dataset, (50, range(10), 5))


def _count_collapses(
    values: tuple[float, ...],
    *,
    task: str,
    base_rate: float,
    lower_is_better: bool,
    metric: str,
) -> int:
    """Count collapsed/degenerate seeds among ``values``.

    Classification: seeds at or below a metric-aware constant-prediction floor.
    A collapsed (constant/random) classifier scores ``base_rate`` on
    ``accuracy`` but ``0.5`` on ``roc_auc`` (chance AUC is 0.5 regardless of
    class imbalance), so the floor is ``0.5 + 0.02`` for ``roc_auc`` and
    ``base_rate + 0.02`` for ``accuracy``. Regression: gross bad-side Tukey
    outliers (beyond ``q75 + 3*IQR``).

    :param values: Per-seed metric values.
    :param task: ``binary_classification`` or ``regression``.
    :param base_rate: Majority-class fraction (classification only).
    :param lower_is_better: Metric direction.
    :param metric: Primary metric name; selects the classification collapse
        floor (``roc_auc`` -> 0.5, ``accuracy`` -> ``base_rate``).
    :returns: Number of collapsed seeds.
    """
    arr = np.asarray(values, dtype=np.float64)
    if task == "binary_classification":
        floor = 0.5 if metric == "roc_auc" else base_rate
        return int((arr <= floor + 0.02).sum())
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = q75 - q25
    return int((arr > q75 + 3.0 * iqr).sum())


def run_dataset(
    dataset: str,
    *,
    backend: str = "torch",
    flavors: tuple[tuple[str, bool, bool], ...] = _ALL_FLAVORS,
    n_trials: int | None = None,
    epochs: int = 50,
    n_jobs: int = 1,
    final_seeds: Iterable[int] | None = None,
    n_splits: int | None = None,
    search_seeds: int = 3,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    storage_dir: Path | None = None,
    search_activation: bool = False,
    max_depth: int = 4,
    embed_layers: int = 1,
) -> list[Path]:
    """Search + final_eval each flavor of one dataset; write per-flavor JSON.

    Budget falls back to `_budget_for(dataset)` when not overridden — the
    per-dataset `_BUDGET` defaults, or the `_SYNTH_BUDGET` preset for `synth_*`.
    Returns the written JSON paths.

    :param search_activation: When ``True``, sample ``activation`` from
        ``{"relu", "elu", "softplus", "selu"}``; otherwise fix it to ``"elu"``.
    :param max_depth: Upper bound of the shallow ``depth`` range (``[1,
        max_depth]``) used when ``deep`` is `False`.
    :param embed_layers: Number of non-monotone `Dense` layers in
        ``cfg.embed_hidden``, each sized ``width``.
    """
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    b_trials, b_seeds, b_splits = _budget_for(dataset)
    n_trials = b_trials if n_trials is None else n_trials
    final_seeds = b_seeds if final_seeds is None else final_seeds
    n_splits = b_splits if n_splits is None else n_splits
    data_dir = data_dir or default_dest()
    out_dir = out_dir or (Path(__file__).resolve().parents[1] / "results" / "phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load(dataset, data_dir=data_dir)
    written: list[Path] = []
    for mode, residual, deep in flavors:
        fname = flavor_name(mode, residual, deep)
        storage = (
            f"sqlite:///{storage_dir}/{dataset}-{fname}.db" if storage_dir else None
        )
        study = search(
            bundle,
            mode=mode,
            residual=residual,
            deep=deep,
            backend=backend,
            n_trials=n_trials,
            epochs=epochs,
            n_jobs=n_jobs,
            n_splits=n_splits,
            search_seeds=search_seeds,
            storage=storage,
            search_activation=search_activation,
            max_depth=max_depth,
            embed_layers=embed_layers,
        )
        agg, eval_rows = final_eval(
            bundle,
            study.best_params,
            mode=mode,
            residual=residual,
            backend=backend,
            seeds=final_seeds,
            epochs=epochs,
            embed_layers=embed_layers,
        )
        base_rate = max(
            float(np.mean(bundle.y_test)), 1.0 - float(np.mean(bundle.y_test))
        )
        rec = {
            "dataset": dataset,
            "flavor": study.flavor,
            "best_params": study.best_params,
            # cv_best is the stability-aware objective (mean -/+ std over
            # folds x search_seeds), not a plain CV mean.
            "cv_best": study.best_value,
            "test_metric": agg.metric,
            # multi-protocol reporting: paper-comparable + outlier-robust.
            "test_mean": agg.mean,
            "test_std": agg.std,
            "test_median": agg.median,
            "test_iqm": agg.iqm,
            "test_values": list(agg.values),
            # Secondary metrics (e.g. accuracy alongside roc_auc); empty for
            # regression datasets, which have a single metric.
            "secondary": _secondary_metrics(eval_rows, agg.metric),
            "n_collapse": _count_collapses(
                agg.values,
                task=bundle.task,
                base_rate=base_rate,
                lower_is_better=_lower_is_better(agg.metric),
                metric=agg.metric,
            ),
            "n_diverged": sum(1 for r in eval_rows if r.diverged),
            "n_seeds": agg.n_seeds,
            "n_train": int(bundle.X_train.shape[0]),
        }
        path = out_dir / f"{dataset}-{fname}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        written.append(path)
    return written
