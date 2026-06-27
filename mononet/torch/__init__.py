"""PyTorch backend for mononet.

Imports `torch` eagerly -- only loaded when the user explicitly
imports `mononet.torch`.
"""

from mononet.torch.layers import MonoInput, MonoLinear, MonoResidual

__all__ = ["MonoInput", "MonoLinear", "MonoResidual"]
