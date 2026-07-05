import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.search import StudyResult, final_eval, flavor_name, search


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + 0.1 * rng.normal(size=120)).astype(np.float64)
    return DatasetBundle(
        name="syn",
        task="regression",
        X_train=X,
        y_train=y,
        X_test=X[:30],
        y_test=y[:30],
        mono_increasing=(0,),
        mono_decreasing=(),
        feature_names=tuple(f"f{i}" for i in range(5)),
        metadata={},
    )


def test_flavor_name() -> None:
    assert flavor_name("switch", False) == "switch-plain"
    assert flavor_name("absolute", True) == "absolute-residual"
    assert flavor_name("switch", True, deep=True) == "switch-deep"
    assert flavor_name("absolute", True, deep=True) == "absolute-deep"


def test_all_flavors_has_six_entries_including_deep() -> None:
    from benchmarks._common.search import _ALL_FLAVORS, flavor_name

    assert len(_ALL_FLAVORS) == 6
    names = {flavor_name(m, r, d) for m, r, d in _ALL_FLAVORS}
    assert {"switch-deep", "absolute-deep"} <= names


def test_search_deep_flavor_names_study_and_uses_high_depth() -> None:
    res = search(
        _bundle(),
        mode="absolute",
        residual=True,
        deep=True,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
        n_splits=2,
    )
    assert res.flavor == "absolute-deep"
    assert res.best_params["depth"] in (6, 10, 16)


def test_search_two_trials_two_folds_returns_finite_best() -> None:
    res = search(
        _bundle(),
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
        n_splits=2,
    )
    assert isinstance(res, StudyResult)
    assert res.n_trials == 2
    assert res.flavor == "switch-plain"
    assert np.isfinite(res.best_value)
    assert "lr" in res.best_params
    assert "width" in res.best_params


def test_search_objective_is_fold_mean() -> None:
    # n_splits=1 (single holdout) and n_splits=3 must both yield a finite CV metric;
    # this exercises the averaging path without asserting an exact value.
    for n_splits in (1, 3):
        res = search(
            _bundle(),
            mode="switch",
            residual=False,
            backend="torch",
            n_trials=2,
            seed=0,
            epochs=1,
            n_splits=n_splits,
        )
        assert np.isfinite(res.best_value)


def test_final_eval_reports_all_seeds() -> None:
    b = _bundle()
    res = search(
        b,
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        n_splits=2,
    )
    # 6 seeds > the old top_k=5 default, so all-seeds reporting is observable:
    # the old best-5-of-6 would give n_selected == 5; the new behaviour gives 6.
    agg = final_eval(
        b,
        res.best_params,
        mode="switch",
        residual=False,
        backend="torch",
        seeds=range(6),
        epochs=1,
    )
    assert np.isfinite(agg.mean)
    assert agg.n_seeds == 6
    assert agg.n_selected == 6  # all seeds reported, no best-k selection
