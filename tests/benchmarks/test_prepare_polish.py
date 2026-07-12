from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.datasets.prepare.polish import (
    MONO_DECREASING,
    MONO_INCREASING,
    prepare_polish,
)


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f"Attr{i}": rng.normal(0, 1, n) for i in range(1, 65)})
    # Inject some missing values (as arff/UCI does) into a few columns,
    # including monotone ones, to exercise median imputation.
    for col in ("Attr1", "Attr2", "Attr35", "Attr50"):
        idx = rng.choice(n, size=5, replace=False)
        df.loc[idx, col] = np.nan
    df["class"] = rng.choice([b"0", b"1"], n)
    return df


def test_prepare_polish_contract() -> None:
    raw = _synthetic_raw()
    train, test = prepare_polish(raw)
    mono = MONO_INCREASING + MONO_DECREASING
    attrs = [f"Attr{i}" for i in range(1, 65)]
    for df in (train, test):
        assert "ground_truth" in df.columns
        assert set(df["ground_truth"].unique()) <= {0, 1}
        assert df.select_dtypes("object").empty  # all numeric
        for col in attrs:
            assert col in df.columns
        for col in mono:
            assert col in df.columns
        assert not df.isna().any().any()  # no NaN anywhere in the output
    assert len(train) == 160  # 80% of 200
    assert len(test) == 40  # 20% of 200
    assert len(set(train.index) & set(test.index)) == 0
    full_ratio = (raw["class"] == b"1").mean()
    assert abs(train["ground_truth"].mean() - full_ratio) < 0.06
    assert abs(test["ground_truth"].mean() - full_ratio) < 0.06
