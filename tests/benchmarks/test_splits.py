import numpy as np

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.splits import train_val_split


def _bundle(task: str, n: int = 200, d: int = 5) -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    if task == "binary_classification":
        y = (X[:, 0] > 0).astype(np.float64)
    else:
        y = X[:, 0] + 0.1 * rng.normal(size=n)
    Xt = rng.normal(size=(10, d))
    yt = np.zeros(10)
    return DatasetBundle(
        name="syn", task=task,  # type: ignore[arg-type]
        X_train=X, y_train=y, X_test=Xt, y_test=yt,
        mono_increasing=(0,), mono_decreasing=(),
        feature_names=tuple(f"f{i}" for i in range(d)), metadata={},
    )


def test_split_sizes_sum_to_train() -> None:
    b = _bundle("regression")
    X_tr, y_tr, X_val, y_val = train_val_split(b, frac=0.2, seed=0)
    assert len(X_tr) + len(X_val) == len(b.X_train)
    assert len(X_val) == 40
    assert len(X_tr) == 160
    assert len(y_tr) == 160
    assert len(y_val) == 40


def test_stratified_preserves_class_balance() -> None:
    b = _bundle("binary_classification")
    _, y_tr, _, y_val = train_val_split(b, frac=0.25, seed=0)
    # both splits contain both classes
    assert set(np.unique(y_tr)) == {0.0, 1.0}
    assert set(np.unique(y_val)) == {0.0, 1.0}


def test_deterministic_for_seed() -> None:
    b = _bundle("regression")
    a = train_val_split(b, seed=3)
    c = train_val_split(b, seed=3)
    assert np.array_equal(a[0], c[0])
    assert np.array_equal(a[2], c[2])


def test_does_not_return_test_data() -> None:
    b = _bundle("regression")
    X_tr, _, X_val, _ = train_val_split(b, seed=0)
    # no row of X_test appears in either split (test is held out entirely)
    train_rows = {r.tobytes() for r in np.vstack([X_tr, X_val])}
    assert not any(r.tobytes() in train_rows for r in b.X_test)
