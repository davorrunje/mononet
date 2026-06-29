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
        final_top_k=2,
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
        "val_best",
        "test_metric",
        "test_mean",
        "test_std",
        "n_seeds",
        "n_selected",
    } <= set(rec)


def test_run_dataset_default_budget_from_table() -> None:
    from benchmarks._common.search import _BUDGET

    assert _BUDGET["auto"][0] == 50
    assert _BUDGET["loan"][0] == 25


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
        storage=storage,
    )
    search(
        bundle,
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        storage=storage,
    )
    study = optuna.load_study(
        study_name=f"auto-{flavor_name('switch', False)}", storage=storage
    )
    assert len(study.trials) == 4  # resumed, not restarted
