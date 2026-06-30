"""Train/validation split helper for hyperparameter search (test untouched)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle


def train_val_split(
    bundle: DatasetBundle,
    *,
    frac: float = 0.2,
    seed: int,
    stratify: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split `bundle`'s train arrays into train/validation.

    :param frac: validation fraction of the train set.
    :param seed: deterministic split seed.
    :param stratify: stratify on `y`; defaults to True for binary classification.
    :returns: `(X_tr, y_tr, X_val, y_val)`. `bundle.X_test`/`y_test` are never read.
    """
    if stratify is None:
        stratify = bundle.task == "binary_classification"
    strat = bundle.y_train if stratify else None
    x_tr, x_val, y_tr, y_val = train_test_split(
        bundle.X_train,
        bundle.y_train,
        test_size=frac,
        random_state=seed,
        stratify=strat,
    )
    return x_tr, y_tr, x_val, y_val


def cv_splits(
    bundle: DatasetBundle,
    *,
    n_splits: int = 5,
    seed: int,
    stratify: bool | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation index folds over `bundle`'s train arrays.

    :param n_splits: number of folds; `>= 2` uses K-fold, `== 1` returns a single
        80/20 holdout (same fraction/stratify default as `train_val_split`).
    :param seed: deterministic shuffling/splitting seed.
    :param stratify: stratify on `y`; defaults to True for binary classification.
    :returns: list of `(train_idx, val_idx)` integer-index arrays into the train
        rows. `bundle.X_test`/`y_test` are never read.
    :raises ValueError: if `n_splits < 1`.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if stratify is None:
        stratify = bundle.task == "binary_classification"
    n = len(bundle.X_train)
    idx = np.arange(n)
    if n_splits == 1:
        strat = bundle.y_train if stratify else None
        tr, val = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=strat
        )
        return [(tr, val)]
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    return [(tr, val) for tr, val in splitter.split(idx, bundle.y_train)]
