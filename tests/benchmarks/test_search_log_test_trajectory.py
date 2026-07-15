import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.search import search
from benchmarks.datasets.download import default_dest
from benchmarks.datasets.registry import load


@pytest.mark.slow
def test_log_test_trajectory_sets_user_attr(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import optuna

    bundle = load("heart", data_dir=default_dest())
    storage = f"sqlite:///{tmp_path}/heart-split-plain.db"
    res = search(
        bundle,
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
    study = optuna.load_study(study_name="heart-split-plain", storage=storage)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert done
    assert all(isinstance(t.user_attrs.get("test_metric"), float) for t in done)
