"""Prepare the UCI Adult (Census Income) dataset in mononet convention.

Monotone (increasing) in ``education_num``, ``hours_per_week``, ``capital_gain``;
``SEX``/``RACE`` are deliberately NOT constrained. Categorical columns are
one-hot encoded; the binary income target becomes ``ground_truth`` (1 = >50K).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

MONO_INCREASING: tuple[str, ...] = ("education_num", "hours_per_week", "capital_gain")


def prepare_adult(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) numeric frames with a fixed stratified 80/20 split.

    :param raw: Raw Adult frame with an ``income`` target column.
    :returns: Train/test frames; target is ``ground_truth`` (0/1), the monotone
        ordinal columns are preserved, categoricals are one-hot encoded.
    """
    df = raw.copy()
    df["ground_truth"] = (df.pop("income").astype(str).str.contains(">50K")).astype(int)
    cat = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and c not in MONO_INCREASING
    ]
    df = pd.get_dummies(df, columns=cat, dtype=int)
    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["ground_truth"]
    )
    # Note: no .reset_index() — the original (disjoint) split indices are kept;
    # downstream writers use index=False, so the index values themselves are
    # never persisted.
    return train, test
