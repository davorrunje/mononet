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

_KNOWN_ACTIVATIONS: frozenset[str] = frozenset(
    {"relu", "elu", "selu", "softplus", "identity"}
)

ActivationName = Literal["relu", "elu", "selu", "softplus", "identity"]


@dataclass(frozen=True, slots=True)
class MonotonicityMask:
    """Per-input-feature monotonicity specification.

    :param values: 1-D array of per-feature signs, each `+1` (output
        non-decreasing in this input) or `-1` (output non-increasing).
        Coerced to `int8`.
    :raises ValueError: If `values` is not 1-D or contains a value outside
        `{-1, +1}`.
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
        """Shape of the underlying mask array.

        :returns: The shape tuple of the `int8` mask array.
        """
        return self.values.shape

    def __len__(self) -> int:
        """Return the number of input features covered by this mask.

        :returns: The number of input features (length of the mask).
        """
        return int(self.values.shape[0])


@dataclass(frozen=True, slots=True)
class ActivationSpec:
    """Backend-agnostic activation specification.

    Backends resolve `name` to their own activation function.

    :param name: Activation name — one of `relu`, `elu`, `selu`, `softplus`,
        `identity`.
    :raises ValueError: If `name` is not a known activation.
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

    :param scheme: Initializer scheme — `he_normal` (default),
        `glorot_uniform`, or `lecun_normal`.
    :param seed: Optional RNG seed for reproducible initialization.
    """

    scheme: Literal["he_normal", "glorot_uniform", "lecun_normal"] = "he_normal"
    seed: int | None = None
