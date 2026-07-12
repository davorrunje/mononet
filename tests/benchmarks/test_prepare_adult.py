from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.datasets.prepare.adult import MONO_INCREASING, prepare_adult


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "education_num": rng.integers(1, 16, n),
            "hours_per_week": rng.integers(1, 80, n),
            "capital_gain": rng.integers(0, 5000, n),
            "workclass": rng.choice(["Private", "Gov"], n),
            "income": rng.choice([">50K", "<=50K"], n),
        }
    )


def test_prepare_adult_contract() -> None:
    train, test = prepare_adult(_synthetic_raw())
    for df in (train, test):
        assert "ground_truth" in df.columns
        assert set(df["ground_truth"].unique()) <= {0, 1}
        assert df.select_dtypes("object").empty  # all numeric
        for col in MONO_INCREASING:
            assert col in df.columns
    assert len(train) == 160  # 80% of 200
    assert len(test) == 40  # 20% of 200
    assert len(set(train.index) & set(test.index)) == 0
