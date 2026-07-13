"""Tests for dataset loaders, specs, and registry."""

from pathlib import Path

from benchmarks.datasets.registry import DATASETS, load

FIXTURES = Path(__file__).parent / "fixtures"


def test_auto_loader_shapes_and_monotonicity() -> None:
    b = load("auto", data_dir=FIXTURES)
    assert b.task == "regression"
    assert b.X_train.shape[1] == len(b.feature_names)
    # weight/displacement/horsepower declared decreasing, none increasing
    assert b.mono_decreasing
    assert not b.mono_increasing
    assert b.X_test.shape[1] == b.X_train.shape[1]


def test_registry_lists_ten_csv_datasets() -> None:
    csv_backed = {k for k, v in DATASETS.items() if v.generator is None}
    assert csv_backed == {
        "adult",
        "auto",
        "blog",
        "compas",
        "heart",
        "loan",
        "taiwan",
        "polish",
        "german",
        "lc",
    }


def test_registry_lists_twelve_synthetic_datasets() -> None:
    synth_keys = {k for k in DATASETS if k.startswith("synth_")}
    assert synth_keys == {
        f"synth_{family}_c{level}"
        for family in ("additive", "teacher_relu", "teacher_elu", "lattice")
        for level in ("low", "mid", "high")
    }
    assert len(synth_keys) == 12
    assert all(DATASETS[k].generator is not None for k in synth_keys)


def test_load_generator_backed_dataset_ignores_data_dir(tmp_path: Path) -> None:
    b = load("synth_teacher_relu_cmid", data_dir=tmp_path)
    assert b.task == "regression"
    assert b.X_train.shape == (16000, 6)
    assert b.X_test.shape == (4000, 6)
    assert b.mono_increasing == tuple(range(6))
    assert b.mono_decreasing == ()
