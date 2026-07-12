"""Prepare the UCI Taiwan Credit (Default of Credit Card Clients) dataset.

Monotone increasing in the repayment-status columns ``PAY_0``..``PAY_6``
(higher delinquency -> higher default risk); monotone decreasing in
``LIMIT_BAL`` and ``PAY_AMT1``..``PAY_AMT6`` (higher credit limit / recent
payments -> lower default risk). ``SEX``/``EDUCATION``/``MARRIAGE``/``AGE``
and the ``BILL_AMT*`` columns (utilization-confounded) are deliberately NOT
constrained. ``ID`` is dropped; the binary target becomes ``ground_truth``.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

MONO_INCREASING: tuple[str, ...] = (
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
)
MONO_DECREASING: tuple[str, ...] = (
    "LIMIT_BAL",
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6",
)


def prepare_taiwan(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) numeric frames with a fixed stratified 80/20 split.

    :param raw: Raw Taiwan Credit frame with an ``ID`` column and a
        ``default payment next month`` target column.
    :returns: Train/test frames; target is ``ground_truth`` (0/1), ``ID`` is
        dropped, the monotone columns are preserved, and any stray
        categoricals are one-hot encoded.
    """
    df = raw.copy()
    df = df.drop(columns=["ID"], errors="ignore")
    df["ground_truth"] = df.pop("default payment next month").astype(int)
    mono = MONO_INCREASING + MONO_DECREASING
    cat = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and c not in mono
    ]
    df = pd.get_dummies(df, columns=cat, dtype=int)
    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["ground_truth"]
    )
    # Note: no .reset_index() — the original (disjoint) split indices are kept;
    # downstream writers use index=False, so the index values themselves are
    # never persisted.
    return train, test
