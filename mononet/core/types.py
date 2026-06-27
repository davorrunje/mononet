# SPDX-License-Identifier: Apache-2.0
"""Shared types used by all mononet backends.

These dataclasses are deliberately simple value objects — no Pydantic.
Validation runs in `__post_init__`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

_KNOWN_ACTIVATIONS: frozenset[str] = frozenset({"relu", "elu", "selu", "softplus"})

ActivationName = Literal["relu", "elu", "selu", "softplus"]


@dataclass(frozen=True, slots=True)
class MonotonicityMask:
    """Per-input-feature monotonicity specification.

    Each entry in `values` is one of `{-1, +1}`:
    - `+1`: output should be monotonically non-decreasing in this input.
    - `-1`: output should be monotonically non-increasing in this input.
    """

    values: npt.NDArray[np.int8]

    def __post_init__(self) -> None:
        """Validate and normalise the mask array."""
        arr = np.asarray(self.values, dtype=np.int8)
        if arr.ndim != 1:
            raise ValueError(f"MonotonicityMask must be 1-D; got shape {arr.shape}")
        if not np.isin(arr, (-1, 1)).all():
            raise ValueError(
                "MonotonicityMask values must be in {-1, +1}; "
                f"got unique values {np.unique(arr).tolist()}"
            )
        # frozen dataclass — assign through object.__setattr__
        object.__setattr__(self, "values", arr)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying mask array."""
        return self.values.shape

    def __len__(self) -> int:
        """Return the number of input features covered by this mask."""
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class ActivationSpec:
    """Backend-agnostic activation specification.

    Backends resolve `name` to their own activation function.
    """

    name: ActivationName

    def __post_init__(self) -> None:
        """Validate that the activation name is known."""
        if self.name not in _KNOWN_ACTIVATIONS:
            raise ValueError(
                f"unknown activation {self.name!r}; known: {sorted(_KNOWN_ACTIVATIONS)}"
            )


@dataclass(frozen=True, slots=True)
class InitSpec:
    """Weight initialization specification.

    Backends resolve `scheme` to their own initializer.
    """

    scheme: Literal["he_normal", "glorot_uniform", "lecun_normal"] = "he_normal"
    seed: int | None = None
