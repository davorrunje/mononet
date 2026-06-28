"""Train/validation split helper for hyperparameter search (test untouched)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import numpy as np

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
        bundle.X_train, bundle.y_train,
        test_size=frac, random_state=seed, stratify=strat,
    )
    return x_tr, y_tr, x_val, y_val
