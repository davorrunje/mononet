"""Canonical dataset container shared by all benchmark loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A preprocessed dataset with declared monotonicity.

    :param mono_increasing: column indices the target is non-decreasing in.
    :param mono_decreasing: column indices the target is non-increasing in.
    """

    name: str
    task: Literal["binary_classification", "regression"]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mono_increasing: tuple[int, ...]
    mono_decreasing: tuple[int, ...]
    feature_names: tuple[str, ...]
    metadata: dict[str, str]


def mono_columns(bundle: DatasetBundle) -> tuple[int, ...]:
    """Monotone column indices, increasing first then decreasing."""
    return (*bundle.mono_increasing, *bundle.mono_decreasing)


def free_columns(bundle: DatasetBundle) -> tuple[int, ...]:
    """Non-monotone column indices, in original order."""
    mono = set(mono_columns(bundle))
    return tuple(i for i in range(len(bundle.feature_names)) if i not in mono)


def mono_signs(bundle: DatasetBundle) -> np.ndarray:
    """`{-1,+1}` int8 sign per monotone column (`+1` increasing, `-1` decreasing)."""
    signs = [1] * len(bundle.mono_increasing) + [-1] * len(bundle.mono_decreasing)
    return np.array(signs, dtype=np.int8)
