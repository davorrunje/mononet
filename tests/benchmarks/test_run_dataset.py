import json
from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.search import run_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def test_run_dataset_writes_one_json_per_flavor(tmp_path: Path) -> None:
    paths = run_dataset(
        "auto",
        backend="torch",
        flavors=(("switch", False), ("absolute", False)),
        n_trials=2,
        epochs=1,
        final_seeds=range(2),
        n_splits=2,
        data_dir=FIXTURES,
        out_dir=tmp_path,
    )
    assert len(paths) == 2
    assert {p.name for p in paths} == {
        "auto-switch-plain.json",
        "auto-absolute-plain.json",
    }
    rec = json.loads(paths[0].read_text())
    assert rec["dataset"] == "auto"
    assert {
        "flavor",
        "best_params",
        "cv_best",
        "test_metric",
        "test_mean",
        "test_std",
        "n_seeds",
    } <= set(rec)
    assert "n_selected" not in rec
    assert "val_best" not in rec


def test_run_dataset_default_budget_from_table() -> None:
    from benchmarks._common.search import _BUDGET

    assert _BUDGET["auto"] == (50, range(10), 5)
    assert _BUDGET["loan"] == (25, range(5), 1)
    assert _BUDGET["blog"][2] == 1  # large datasets use single holdout


def test_storage_uses_deterministic_study_name_so_it_resumes(tmp_path: Path) -> None:
    # With a fixed study_name + shared storage, a second search RESUMES the same
    # study (accumulating trials) rather than starting a new one. This is what
    # makes --storage-dir resumable and same-study multi-worker concurrency possible.
    import optuna

    from benchmarks._common.search import flavor_name, search
    from benchmarks.datasets.registry import load

    bundle = load("auto", data_dir=FIXTURES)
    storage = f"sqlite:///{tmp_path}/auto.db"
    search(
        bundle,
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        n_splits=2,
        storage=storage,
    )
    search(
        bundle,
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        n_splits=2,
        storage=storage,
    )
    study = optuna.load_study(
        study_name=f"auto-{flavor_name('switch', False)}", storage=storage
    )
    assert len(study.trials) == 4  # resumed, not restarted
