from __future__ import annotations

import numpy as np

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.splits import subsample_train


def _bundle(n: int = 400, pos_frac: float = 0.3) -> DatasetBundle:
    rng = np.random.default_rng(0)
    y = (rng.random(n) < pos_frac).astype(np.float64)
    x = rng.standard_normal((n, 4))
    yt = (rng.random(80) < pos_frac).astype(np.float64)
    return DatasetBundle(
        name="synthetic",
        task="binary_classification",
        X_train=x,
        y_train=y,
        X_test=rng.standard_normal((80, 4)),
        y_test=yt,
        mono_increasing=(0, 1),
        mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"),
        metadata={},
    )


def test_subsample_size_and_test_untouched() -> None:
    """Subsample yields exactly n train rows and leaves test arrays identical."""
    b = _bundle()
    s = subsample_train(b, 100, seed=0)
    assert len(s.X_train) == 100
    assert len(s.y_train) == 100
    assert np.array_equal(s.X_test, b.X_test)
    assert np.array_equal(s.y_test, b.y_test)


def test_subsample_preserves_class_ratio() -> None:
    """Stratified subsample allocates the positive class near-exactly.

    Asserts ±1 row of the expected count for *every* seed — a bound an
    unstratified draw (binomial noise ≈ ±4 here) would violate on some seed,
    so this actually exercises the stratified path.
    """
    b = _bundle(pos_frac=0.3)
    expected_pos = 100 * float(b.y_train.mean())
    for seed in range(10):
        s = subsample_train(b, 100, seed=seed)
        assert abs(float(s.y_train.sum()) - expected_pos) <= 1.0


def test_subsample_deterministic_and_seed_varies() -> None:
    """Same seed → identical rows; different seed → different rows."""
    b = _bundle()
    a0 = subsample_train(b, 100, seed=0)
    a0b = subsample_train(b, 100, seed=0)
    a1 = subsample_train(b, 100, seed=1)
    assert np.array_equal(a0.X_train, a0b.X_train)
    assert not np.array_equal(a0.X_train, a1.X_train)


def test_subsample_full_returns_unchanged() -> None:
    """N >= train size returns the same bundle object."""
    b = _bundle(n=400)
    assert subsample_train(b, 400, seed=0) is b
    assert subsample_train(b, 999, seed=0) is b
