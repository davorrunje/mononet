# SPDX-License-Identifier: Apache-2.0
"""mononet — Unconstrained monotonic neural networks.

Multi-backend support for PyTorch, JAX (Flax NNX), and Keras 3.
See https://arxiv.org/abs/2205.11775 for the reference paper.

Backends are imported lazily: `import mononet` does **not** import
torch / jax / keras. Use `from mononet.torch import ...` (or the
equivalent for jax/keras) to access backend layers.
"""

from importlib.metadata import version

from mononet.core.types import MonotonicityMask

__version__ = version("mononet")

__all__ = ["MonotonicityMask", "__version__"]
