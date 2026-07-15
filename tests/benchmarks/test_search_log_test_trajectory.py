from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.search import search


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


@pytest.mark.slow
def test_log_test_trajectory_sets_user_attr(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import optuna

    storage = f"sqlite:///{tmp_path}/syn-split-plain.db"
    res = search(
        _bundle(),
        mode="split",
        residual=False,
        backend="torch",
        n_trials=2,
        n_splits=2,
        search_seeds=1,
        epochs=2,
        embed_layers=2,
        storage=storage,
        log_test_trajectory=True,
    )
    assert res.n_trials == 2
    study = optuna.load_study(study_name="syn-split-plain", storage=storage)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert done
    assert all(isinstance(t.user_attrs.get("test_metric"), float) for t in done)
