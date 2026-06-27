"""Tests for dataset loaders, specs, and registry."""

from pathlib import Path

from benchmarks.datasets.registry import DATASETS, load

FIXTURES = Path(__file__).parent / "fixtures"


def test_auto_loader_shapes_and_monotonicity():
    b = load("auto", data_dir=FIXTURES)
    assert b.task == "regression"
    assert b.X_train.shape[1] == len(b.feature_names)
    # weight/displacement/horsepower declared decreasing, none increasing
    assert b.mono_decreasing
    assert not b.mono_increasing
    assert b.X_test.shape[1] == b.X_train.shape[1]


def test_registry_lists_five_datasets():
    assert set(DATASETS) == {"auto", "blog", "compas", "heart", "loan"}
