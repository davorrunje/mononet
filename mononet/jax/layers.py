"""JAX (Flax NNX) idiomatic layer wrappers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flax import nnx

if TYPE_CHECKING:
    import jax.numpy as jnp

    from mononet.core.types import ActivationSpec, MonotonicityMask


class MonoLinear(nnx.Module):
    """Monotonic analogue of `flax.nnx.Linear`."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise MonoLinear (not yet implemented)."""
        raise NotImplementedError("MonoLinear JAX wrapper lands in the follow-up plan.")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:  # pragma: no cover
        """Apply the monotonic linear transformation."""
        raise NotImplementedError
