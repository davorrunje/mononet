from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.datasets.prepare.german import (
    MONO_DECREASING,
    MONO_INCREASING,
    prepare_german,
)


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "checking_status": rng.choice(["A11", "A12", "A13", "A14"], n),
            "duration": rng.integers(4, 72, n),
            "credit_history": rng.choice(["A30", "A31", "A32", "A33", "A34"], n),
            "purpose": rng.choice(["A40", "A41", "A42", "A43"], n),
            "credit_amount": rng.integers(250, 20_000, n),
            "savings": rng.choice(["A61", "A62", "A63", "A64", "A65"], n),
            "employment": rng.choice(["A71", "A72", "A73", "A74", "A75"], n),
            "installment_rate": rng.integers(1, 5, n),
            "personal_status_sex": rng.choice(["A91", "A92", "A93", "A94"], n),
            "other_debtors": rng.choice(["A101", "A102", "A103"], n),
            "residence_since": rng.integers(1, 5, n),
            "property": rng.choice(["A121", "A122", "A123", "A124"], n),
            "age": rng.integers(19, 75, n),
            "other_installment_plans": rng.choice(["A141", "A142", "A143"], n),
            "housing": rng.choice(["A151", "A152", "A153"], n),
            "existing_credits": rng.integers(1, 4, n),
            "job": rng.choice(["A171", "A172", "A173", "A174"], n),
            "people_liable": rng.integers(1, 3, n),
            "telephone": rng.choice(["A191", "A192"], n),
            "foreign_worker": rng.choice(["A201", "A202"], n),
            "target": rng.choice([1, 2], n),
        }
    )
    return df


def test_prepare_german_contract() -> None:
    raw = _synthetic_raw()
    train, test = prepare_german(raw)
    mono = MONO_INCREASING + MONO_DECREASING
    for df in (train, test):
        assert "ground_truth" in df.columns
        assert set(df["ground_truth"].unique()) <= {0, 1}
        assert df.select_dtypes("object").empty  # all numeric
        assert "target" not in df.columns
        for col in mono:
            assert col in df.columns
    assert len(train) == 160  # 80% of 200
    assert len(test) == 40  # 20% of 200
    assert len(set(train.index) & set(test.index)) == 0
    # Stratified split preserves the overall positive-class (bad=1) ratio.
    full_ratio = (raw["target"] == 2).mean()
    assert abs(train["ground_truth"].mean() - full_ratio) < 0.06
    assert abs(test["ground_truth"].mean() - full_ratio) < 0.06
