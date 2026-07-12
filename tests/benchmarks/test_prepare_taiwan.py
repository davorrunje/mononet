from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.datasets.prepare.taiwan import (
    MONO_DECREASING,
    MONO_INCREASING,
    prepare_taiwan,
)


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ID": np.arange(1, n + 1)})
    df["LIMIT_BAL"] = rng.integers(10_000, 500_000, n)
    df["SEX"] = rng.integers(1, 3, n)
    df["EDUCATION"] = rng.integers(1, 5, n)
    df["MARRIAGE"] = rng.integers(1, 4, n)
    df["AGE"] = rng.integers(21, 70, n)
    for i in (0, 2, 3, 4, 5, 6):
        df[f"PAY_{i}"] = rng.integers(-2, 9, n)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.integers(-10_000, 50_000, n)
    for i in range(1, 7):
        df[f"PAY_AMT{i}"] = rng.integers(0, 20_000, n)
    df["default payment next month"] = rng.choice([0, 1], n)
    return df


def test_prepare_taiwan_contract() -> None:
    raw = _synthetic_raw()
    train, test = prepare_taiwan(raw)
    mono = MONO_INCREASING + MONO_DECREASING
    for df in (train, test):
        assert "ground_truth" in df.columns
        assert set(df["ground_truth"].unique()) <= {0, 1}
        assert df.select_dtypes("object").empty  # all numeric
        assert "ID" not in df.columns
        for col in mono:
            assert col in df.columns
    assert len(train) == 160  # 80% of 200
    assert len(test) == 40  # 20% of 200
    assert len(set(train.index) & set(test.index)) == 0
    # Stratified split preserves the overall positive-class ratio.
    full_ratio = raw["default payment next month"].mean()
    assert abs(train["ground_truth"].mean() - full_ratio) < 0.05
    assert abs(test["ground_truth"].mean() - full_ratio) < 0.05
