# SPDX-License-Identifier: Apache-2.0
"""Problem registry: one plug-in module per PDE family.

Importing this package registers the built-in problems (conservation laws).
Follow-up papers add ``hjb`` / ``fokker_planck`` / ``eikonal`` modules here.
"""

from __future__ import annotations

from applications.pinn.core.problems import conservation  # noqa: F401  (registers)
from applications.pinn.core.problems.base import (
    Problem,
    available,
    get,
    register,
)

__all__ = ["Problem", "available", "get", "register"]
