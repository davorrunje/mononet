"""Optuna search engine over the Phase-1 run() harness (validation-driven)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import optuna

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.results import Aggregate, aggregate
from benchmarks._common.runner import run
from benchmarks._common.search_spaces import suggest_config
from benchmarks._common.splits import train_val_split

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


def _val_bundle(bundle: DatasetBundle, seed: int) -> DatasetBundle:
    """Throwaway bundle with the held-out validation set in the test slot.

    Lets the search reuse run() (which evaluates on X_test) to score on
    validation without ever touching the real test set.
    """
    x_tr, y_tr, x_val, y_val = train_val_split(bundle, seed=seed)
    return dataclasses.replace(
        bundle, X_train=x_tr, y_train=y_tr, X_test=x_val, y_test=y_val
    )


def search(
    bundle: DatasetBundle,
    *,
    mode: str,
    residual: bool,
    backend: str,
    n_trials: int = 50,
    seed: int = 0,
    epochs: int = 50,
    metric: str | None = None,
    storage: str | None = None,
) -> StudyResult:
    """Tune (dataset, flavor) HPs on a validation split via Optuna TPE."""
    metric = metric or _primary_metric(bundle)
    direction = "minimize" if _lower_is_better(metric) else "maximize"
    vb = _val_bundle(bundle, seed)

    def objective(trial: optuna.Trial) -> float:
        cfg: BenchmarkConfig = suggest_config(
            trial, dataset=bundle.name, backend=backend,  # type: ignore[arg-type]
            mode=mode, residual=residual, epochs=epochs,  # type: ignore[arg-type]
        )
        rows = run(cfg, vb)
        return float(rows[0].scores[metric])  # type: ignore[index]

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials)
    return StudyResult(
        dataset=bundle.name, flavor=flavor_name(mode, residual),
        best_params=dict(study.best_params), best_value=float(study.best_value),
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
    top_k: int = 5,
) -> Aggregate:
    """Refit best HPs on the full train split; report TEST best-k-of-n."""
    metric = metric or _primary_metric(bundle)
    width = int(best_params["width"])
    cfg = BenchmarkConfig(
        dataset=bundle.name, backend=backend,  # type: ignore[arg-type]
        mode=mode, residual=residual,  # type: ignore[arg-type]
        depth=int(best_params["depth"]), width=width, activation="elu",
        convex_fraction=float(best_params.get("convex_fraction", 0.5)),
        embed_hidden=(width,), dropout=float(best_params["dropout"]),
        optimizer=OptimizerSpec(
            "adam", float(best_params["lr"]), float(best_params["weight_decay"])
        ),
        lr_decay=float(best_params["lr_decay"]), batch_size=int(best_params["batch_size"]),
        epochs=epochs, early_stopping=None, seeds=tuple(seeds),
        metrics=(metric,),  # type: ignore[arg-type]
    )
    rows = run(cfg, bundle)
    return aggregate(
        rows, metric=metric, lower_is_better=_lower_is_better(metric), top_k=top_k
    )
