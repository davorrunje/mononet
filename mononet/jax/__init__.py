"""JAX backend (Flax NNX) for mononet."""

from mononet.jax.layers import MonoLinear
from mononet.jax.models import MonoMLP

__all__ = ["MonoLinear", "MonoMLP"]
