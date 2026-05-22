"""Keras monotonic-model compositions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import keras

if TYPE_CHECKING:
    from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoMLP(keras.Model):  # type: ignore[misc]
    """Multi-layer monotonic MLP, Keras backend."""

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        **kwargs: Any,
    ) -> None:
        """Initialise MonoMLP (not yet implemented)."""
        super().__init__(**kwargs)
        raise NotImplementedError(
            "MonoMLP Keras composition lands in the follow-up plan."
        )

    def call(self, inputs: Any) -> Any:  # pragma: no cover
        """Apply the monotonic MLP."""
        raise NotImplementedError
