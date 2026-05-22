"""JAX monotonic-model compositions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

if TYPE_CHECKING:
    import jax.numpy as jnp

    from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoMLP(nnx.Module):
    """Multi-layer monotonic MLP, JAX backend."""

    def __init__(
        self,
        in_features: int,
        hidden_features: list[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise MonoMLP (not yet implemented)."""
        raise NotImplementedError(
            "MonoMLP JAX composition lands in the follow-up plan."
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover
        """Apply the monotonic MLP."""
        raise NotImplementedError
