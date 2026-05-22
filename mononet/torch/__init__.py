"""PyTorch backend for mononet.

Imports `torch` eagerly — only loaded when the user explicitly
imports `mononet.torch`.
"""

from mononet.torch.layers import MonoLinear
from mononet.torch.models import MonoMLP

__all__ = ["MonoLinear", "MonoMLP"]
