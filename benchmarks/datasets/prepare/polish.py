"""Prepare the UCI Polish Companies Bankruptcy dataset (3-year horizon).

All 64 ``Attr1``..``Attr64`` features are kept (all numeric); missing values
(arriving as NaN once the raw ARFF byte target is decoded) are median-imputed
per column for determinism. Only 6 of the 64 are constrained: monotone
increasing in ``Attr2`` (liabilities/assets — higher leverage -> higher
bankruptcy risk); monotone decreasing in ``Attr1`` (ROA), ``Attr4`` (current
ratio), ``Attr17`` (assets/liabilities), ``Attr23`` (net margin), ``Attr35``
(profit-on-sales/assets) — higher profitability/liquidity/solvency -> lower
bankruptcy risk. The ``class`` byte target becomes ``ground_truth`` (0/1).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

MONO_INCREASING: tuple[str, ...] = ("Attr2",)
MONO_DECREASING: tuple[str, ...] = ("Attr1", "Attr4", "Attr17", "Attr23", "Attr35")


def prepare_polish(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) numeric frames with a fixed stratified 80/20 split.

    :param raw: Raw Polish bankruptcy frame with ``Attr1``..``Attr64`` feature
        columns and a ``class`` target column (bytes or numeric, coercible to
        0/1).
    :returns: Train/test frames; target is ``ground_truth`` (0/1), all 64
        ``Attr*`` columns are retained and median-imputed per column (no NaN
        remains), and the 6 monotone columns are preserved.
    """
    df = raw.copy()
    target = df.pop("class")
    if pd.api.types.is_object_dtype(target) or isinstance(
        target.iloc[0] if len(target) else b"", bytes
    ):
        target = target.apply(lambda v: v.decode() if isinstance(v, bytes) else v)
    df["ground_truth"] = target.astype(int)

    attrs = [c for c in df.columns if c != "ground_truth"]
    df[attrs] = df[attrs].fillna(df[attrs].median())

    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["ground_truth"]
    )
    # Note: no .reset_index() — the original (disjoint) split indices are kept;
    # downstream writers use index=False, so the index values themselves are
    # never persisted.
    return train, test
