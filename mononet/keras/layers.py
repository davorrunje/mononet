"""Keras 3 idiomatic layer wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import keras

if TYPE_CHECKING:
    from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoDense(keras.layers.Layer):  # type: ignore[misc]
    """Monotonic analogue of `keras.layers.Dense`."""

    def __init__(
        self,
        units: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        **kwargs: Any,
    ) -> None:
        """Initialise MonoDense (not yet implemented)."""
        super().__init__(**kwargs)
        raise NotImplementedError(
            "MonoDense Keras wrapper lands in the follow-up plan."
        )

    def call(self, inputs: Any) -> Any:  # pragma: no cover
        """Apply the monotonic dense transformation."""
        raise NotImplementedError
