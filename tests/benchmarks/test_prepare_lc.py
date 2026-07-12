from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.datasets.prepare.lc import (
    MONO_DECREASING,
    MONO_INCREASING,
    prepare_lc,
)


def _synthetic_raw(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    years = rng.choice([2014, 2015, 2016, 2017, 2018], n)
    months = rng.choice(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
        n,
    )
    df = pd.DataFrame(
        {
            "revenue": rng.uniform(20_000, 150_000, n),
            "dti_n": rng.uniform(0, 40, n),
            "loan_amnt": rng.uniform(1_000, 40_000, n),
            "fico_n": rng.integers(600, 850, n),
            "experience_c": rng.integers(0, 2, n),
            "emp_length": rng.choice(["< 1 year", "1 year", "5 years", "10+ years"], n),
            "purpose": rng.choice(
                ["debt_consolidation", "credit_card", "home_improvement"], n
            ),
            "home_ownership_n": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
            "Default": rng.choice([0, 1], n),
            "issue_d": [f"{m}-{y}" for m, y in zip(months, years, strict=True)],
        }
    )
    return df


def _years(raw: pd.DataFrame, index: pd.Index) -> pd.Series:
    return raw.loc[index, "issue_d"].str.split("-").str[-1].astype(int)


def test_prepare_lc_chronological_split() -> None:
    raw = _synthetic_raw()
    train, test = prepare_lc(raw)
    mono = MONO_INCREASING + MONO_DECREASING

    for df in (train, test):
        assert "ground_truth" in df.columns
        assert set(df["ground_truth"].unique()) <= {0, 1}
        assert df.select_dtypes("object").empty  # all numeric
        assert "issue_d" not in df.columns
        assert "Default" not in df.columns
        for col in mono:
            assert col in df.columns

    assert set(train.columns) == set(test.columns)
    assert len(set(train.index) & set(test.index)) == 0

    train_years = _years(raw, train.index)
    test_years = _years(raw, test.index)
    assert train_years.max() <= 2015
    assert test_years.min() >= 2017
    assert 2016 not in set(train_years)
    assert 2016 not in set(test_years)
    # 2016 rows are excluded entirely from both splits.
    assert len(train) + len(test) < len(raw)
    assert (raw["issue_d"].str.split("-").str[-1].astype(int) == 2016).sum() > 0
