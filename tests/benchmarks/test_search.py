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


def test_search_two_trials_returns_finite_best() -> None:
    res = search(
        _bundle(),
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
    )
    assert isinstance(res, StudyResult)
    assert res.n_trials == 2
    assert res.flavor == "switch-plain"
    assert np.isfinite(res.best_value)
    assert "lr" in res.best_params
    assert "width" in res.best_params


def test_final_eval_returns_aggregate_on_test() -> None:
    b = _bundle()
    res = search(
        b, mode="switch", residual=False, backend="torch", n_trials=2, epochs=1
    )
    agg = final_eval(
        b,
        res.best_params,
        mode="switch",
        residual=False,
        backend="torch",
        seeds=range(2),
        epochs=1,
        top_k=2,
    )
    assert np.isfinite(agg.mean)
    assert agg.n_selected == 2
