"""Private JAX kernels for monotonic primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import jax.numpy as jnp

    from mononet.core.types import ActivationSpec, MonotonicityMask


def monotonic_dense(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> jnp.ndarray:
    """JAX kernel for the monotonic dense transformation."""
    raise NotImplementedError("monotonic_dense JAX kernel lands in the follow-up plan.")
