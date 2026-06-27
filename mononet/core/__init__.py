"""Framework-agnostic primitives shared by all backends."""

from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask

__all__ = [
    "ActivationSpec",
    "InitSpec",
    "MonoConfig",
    "MonoResidualConfig",
    "MonotonicityMask",
]
