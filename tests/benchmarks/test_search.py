from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.search import (
    StudyResult,
    _primary_metric,
    final_eval,
    flavor_name,
    search,
)

if TYPE_CHECKING:
    from pathlib import Path


def _bundle(task: str = "regression") -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    if task == "binary_classification":
        y = (X[:, 0] > 0).astype(np.float64)
    else:
        y = (X[:, 0] + 0.1 * rng.normal(size=120)).astype(np.float64)
    return DatasetBundle(
        name="syn",
        task=task,  # type: ignore[arg-type]
        X_train=X,
        y_train=y,
        X_test=X[:30],
        y_test=y[:30],
        mono_increasing=(0,),
        mono_decreasing=(),
        feature_names=tuple(f"f{i}" for i in range(5)),
        metadata={},
    )


def test_primary_metric_is_roc_auc_for_binary_classification() -> None:
    assert _primary_metric(_bundle(task="binary_classification")) == "roc_auc"


def test_primary_metric_is_mse_for_regression() -> None:
    assert _primary_metric(_bundle(task="regression")) == "mse"


def test_flavor_name() -> None:
    assert flavor_name("split", False) == "split-plain"
    assert flavor_name("mixed", True) == "mixed-residual"
    assert flavor_name("split", True, deep=True) == "split-deep"
    assert flavor_name("mixed", True, deep=True) == "mixed-deep"


def test_all_flavors_has_six_entries_including_deep() -> None:
    from benchmarks._common.search import _ALL_FLAVORS, flavor_name

    assert len(_ALL_FLAVORS) == 6
    names = {flavor_name(m, r, d) for m, r, d in _ALL_FLAVORS}
    assert {"split-deep", "mixed-deep"} <= names


def test_search_deep_flavor_names_study_and_uses_high_depth() -> None:
    res = search(
        _bundle(),
        mode="mixed",
        residual=True,
        deep=True,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
        n_splits=2,
    )
    assert res.flavor == "mixed-deep"
    assert res.best_params["depth"] in (6, 10, 16)


def test_search_two_trials_two_folds_returns_finite_best() -> None:
    res = search(
        _bundle(),
        mode="split",
        residual=False,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
        n_splits=2,
    )
    assert isinstance(res, StudyResult)
    assert res.n_trials == 2
    assert res.flavor == "split-plain"
    assert np.isfinite(res.best_value)
    assert "lr" in res.best_params
    assert "width" in res.best_params


def test_search_objective_is_fold_mean() -> None:
    # n_splits=1 (single holdout) and n_splits=3 must both yield a finite CV metric;
    # this exercises the averaging path without asserting an exact value.
    for n_splits in (1, 3):
        res = search(
            _bundle(),
            mode="split",
            residual=False,
            backend="torch",
            n_trials=2,
            seed=0,
            epochs=1,
            n_splits=n_splits,
        )
        assert np.isfinite(res.best_value)


def test_classification_final_eval_reports_roc_auc_and_accuracy() -> None:
    # With roc_auc as the classification primary, final_eval must still compute
    # and store accuracy alongside it in each ResultRow's scores dict.
    from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
    from benchmarks._common.runner import run

    b = _bundle(task="binary_classification")
    cfg = BenchmarkConfig(
        dataset="syn",
        backend="torch",
        mode="split",
        residual=False,
        depth=1,
        width=8,
        activation="elu",
        convex_fraction=0.5,
        embed_hidden=(8,),
        dropout=0.0,
        optimizer=OptimizerSpec("adam", 1e-2, 0.0),
        lr_decay=None,
        batch_size=32,
        epochs=2,
        early_stopping=None,
        seeds=(0,),
        metrics=("roc_auc", "accuracy"),
    )
    rows = run(cfg, b)
    assert rows
    assert "roc_auc" in rows[0].scores
    assert "accuracy" in rows[0].scores


def test_final_eval_reports_all_seeds() -> None:
    b = _bundle()
    res = search(
        b,
        mode="split",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        n_splits=2,
    )
    # 6 seeds > the old top_k=5 default, so all-seeds reporting is observable:
    # the old best-5-of-6 would give n_selected == 5; the new behaviour gives 6.
    agg, rows = final_eval(
        b,
        res.best_params,
        mode="split",
        residual=False,
        backend="torch",
        seeds=range(6),
        epochs=1,
    )
    assert np.isfinite(agg.mean)
    assert agg.n_seeds == 6
    assert agg.n_selected == 6  # all seeds reported, no best-k selection
    assert len(rows) == 6


def test_run_dataset_persists_secondary_accuracy_for_classification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Classification run_dataset records secondary["accuracy"] alongside roc_auc.

    `final_eval` computes both `roc_auc` (primary) and `accuracy` for every
    classification seed, but the committed rec previously stored only the
    primary metric's Aggregate — dropping accuracy from the persisted JSON.
    """
    from benchmarks._common.search import run_dataset

    bundle = _bundle(task="binary_classification")

    def fake_load(name: str, *, data_dir: Any) -> DatasetBundle:
        return bundle

    monkeypatch.setattr("benchmarks.datasets.registry.load", fake_load)

    paths = run_dataset(
        "syn",
        backend="torch",
        flavors=(("split", False, False),),
        n_trials=2,
        epochs=1,
        final_seeds=range(3),
        n_splits=2,
        out_dir=tmp_path,
    )
    rec = json.loads(paths[0].read_text())
    assert rec["test_metric"] == "roc_auc"
    sec = rec["secondary"]["accuracy"]
    assert np.isfinite(sec["iqm"])
    assert len(sec["values"]) == 3


def test_run_dataset_persists_n_diverged(tmp_path: Path) -> None:
    """run_dataset must record n_diverged alongside n_collapse in the JSON.

    Uses the generator-backed `synth_additive_clow` dataset (no data files
    needed) end-to-end through search + final_eval + JSON persistence.
    """
    from benchmarks._common.search import run_dataset

    paths = run_dataset(
        "synth_additive_clow",
        backend="torch",
        flavors=(("mixed", False, False),),
        n_trials=1,
        epochs=2,
        final_seeds=range(2),
        n_splits=2,
        search_seeds=1,
        out_dir=tmp_path,
    )
    rec = json.loads(paths[0].read_text())
    assert (tmp_path / "synth_additive_clow-mixed-plain.json") in paths
    assert "n_diverged" in rec
    assert isinstance(rec["n_diverged"], int)
    assert 0 <= rec["n_diverged"] <= rec["n_seeds"]


def _tiny_reg_bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (80, 3)).astype("float32")
    y = x.sum(1).astype("float32")
    return DatasetBundle(
        name="t",
        task="regression",
        X_train=x,
        y_train=y,
        X_test=x,
        y_test=y,
        mono_increasing=(0, 1, 2),
        mono_decreasing=(),
        feature_names=("a", "b", "c"),
        metadata={},
    )


def test_final_eval_honors_activation_and_alt_init() -> None:
    b = _tiny_reg_bundle()
    params = {
        "width": 8,
        "depth": 2,
        "dropout": 0.0,
        "lr": 1e-2,
        "weight_decay": 0.0,
        "lr_decay": 1.0,
        "batch_size": 32,
        "activation": "relu",
    }
    _agg, rows = final_eval(
        b,
        params,
        mode="alternate",
        residual=False,
        backend="torch",
        seeds=range(2),
        epochs=2,
        embed_layers=2,
    )
    assert len(rows) == 2
    assert all("mse" in r.scores for r in rows)


def test_search_alternate_shallow_runs() -> None:
    b = _tiny_reg_bundle()
    res = search(
        b,
        mode="alternate",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=2,
        n_splits=2,
        search_seeds=1,
        search_activation=True,
        max_depth=3,
        embed_layers=2,
    )
    assert res.flavor == "alternate-plain"
    assert 1 <= res.best_params["depth"] <= 3
    assert res.best_params["activation"] in ("relu", "elu", "softplus", "selu")
