import numpy as np

from benchmarks._common.bundle import (
    DatasetBundle,
    free_columns,
    mono_columns,
    mono_signs,
)


def _bundle() -> DatasetBundle:
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    y = np.array([0.0, 1.0, 0.0])
    return DatasetBundle(
        name="t", task="binary_classification",
        X_train=X, y_train=y, X_test=X, y_test=y,
        mono_increasing=(0,), mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"), metadata={},
    )


def test_mono_columns_increasing_then_decreasing():
    b = _bundle()
    assert mono_columns(b) == (0, 2)
    assert free_columns(b) == (1, 3)


def test_mono_signs_match_direction():
    b = _bundle()
    assert mono_signs(b).tolist() == [1, -1]
    assert mono_signs(b).dtype == np.int8


def test_bundle_is_frozen():
    b = _bundle()
    try:
        b.name = "x"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("DatasetBundle must be frozen")
