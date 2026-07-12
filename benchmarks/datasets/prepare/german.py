"""Prepare the UCI Statlog German Credit dataset.

Monotone increasing in ``duration``, ``credit_amount``, ``installment_rate``
(higher -> higher default risk); monotone decreasing in ``age`` (higher ->
lower default risk). All coded categorical attributes (``checking_status``,
``credit_history``, ``purpose``, ``savings``, ``employment``,
``personal_status_sex``, ``other_debtors``, ``property``,
``other_installment_plans``, ``housing``, ``job``, ``telephone``,
``foreign_worker`` — values like ``A11``/``A34``) are one-hot encoded; the
numeric non-monotone columns (``residence_since``, ``existing_credits``,
``people_liable``) are kept raw but unconstrained. The coded target
(``1=good, 2=bad``) becomes ``ground_truth`` (0/1, 1 = bad/default).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

#: Column names for the raw space-separated, header-less UCI file, in order
#: (20 features + the 21st, ``target``).
COLUMNS: tuple[str, ...] = (
    "checking_status",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "people_liable",
    "telephone",
    "foreign_worker",
    "target",
)

MONO_INCREASING: tuple[str, ...] = ("duration", "credit_amount", "installment_rate")
MONO_DECREASING: tuple[str, ...] = ("age",)


def prepare_german(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) numeric frames with a fixed stratified 80/20 split.

    :param raw: Raw German Credit frame with the 20 documented feature
        columns (coded categoricals as ``A##`` strings) and a ``target``
        column coded ``1=good, 2=bad``.
    :returns: Train/test frames; target is ``ground_truth`` (0/1, 1 = bad),
        the 4 monotone columns are preserved raw, and the coded categoricals
        are one-hot encoded.
    """
    df = raw.copy()
    df["ground_truth"] = (df.pop("target") == 2).astype(int)
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
