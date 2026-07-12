"""Prepare the Lending Club-Zenodo loan-granting dataset (Zenodo 11295916).

Monotone increasing in ``dti_n`` (higher debt-to-income -> higher default
risk); monotone decreasing in ``fico_n`` and ``revenue`` (higher credit score
/ income -> lower default risk). These are the 3 columns the source paper
constrains; ``loan_amnt`` is kept as a raw numeric feature but deliberately
left unconstrained. The categoricals ``emp_length``, ``purpose``,
``home_ownership_n`` are one-hot encoded; ``experience_c`` is a binary flag
coerced to ``int``. The ``Default`` target becomes ``ground_truth`` (0/1,
1 = default).

The split is **chronological**, not random/stratified: ``issue_d`` (the loan
issue date, e.g. ``"Dec-2015"``) is parsed to a calendar year. Train keeps
loans issued 2007-2015, test keeps loans issued 2017-2018, and year 2016 is
excluded from both splits entirely (a one-year gap between train and test so
the boundary isn't blurred by loans whose outcome only resolves after
``issue_d``). ``issue_d`` itself is dropped after computing the split — it is
not a model feature.
"""

from __future__ import annotations

import pandas as pd

MONO_INCREASING: tuple[str, ...] = ("dti_n",)
MONO_DECREASING: tuple[str, ...] = ("fico_n", "revenue")

#: Train keeps loans issued in [2007, _TRAIN_MAX_YEAR]; test keeps loans
#: issued in [_TEST_MIN_YEAR, 2018]. Year 2016 (the gap) is in neither.
_TRAIN_MAX_YEAR = 2015
_TEST_MIN_YEAR = 2017


def _issue_year(issue_d: pd.Series) -> pd.Series:
    """Parse the 4-digit calendar year out of ``issue_d`` strings.

    :param issue_d: Series of ``"Mon-YYYY"`` strings, e.g. ``"Dec-2015"``.
    :returns: Integer year series, same index as *issue_d*.
    """
    return issue_d.astype(str).str.split("-").str[-1].astype(int)


def prepare_lc(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) numeric frames with a fixed CHRONOLOGICAL split.

    :param raw: Raw frame with the 8 documented feature columns
        (``revenue``, ``dti_n``, ``loan_amnt``, ``fico_n``, ``experience_c``,
        ``emp_length``, ``purpose``, ``home_ownership_n``), a ``Default``
        target column (0/1, 1 = default), and an ``issue_d`` loan-issue-date
        column (e.g. ``"Dec-2015"``) used only to compute the split.
    :returns: Train (issued 2007-2015) / test (issued 2017-2018) frames;
        ``issue_d`` and ``Default`` are dropped/renamed, target is
        ``ground_truth`` (0/1), the 3 monotone columns are preserved raw,
        and the categoricals are one-hot encoded.
    """
    df = raw.copy()
    year = _issue_year(df.pop("issue_d"))
    df["ground_truth"] = df.pop("Default").astype(int)
    df["experience_c"] = df["experience_c"].astype(int)

    cat = ["emp_length", "purpose", "home_ownership_n"]
    df = pd.get_dummies(df, columns=cat, dtype=int)

    train = df[year <= _TRAIN_MAX_YEAR]
    test = df[year >= _TEST_MIN_YEAR]
    # Note: no .reset_index() — the original (disjoint) split indices are kept;
    # downstream writers use index=False, so the index values themselves are
    # never persisted.
    return train, test
