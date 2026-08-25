# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic science for the PINN application (pure NumPy).

Single source of truth: problem definitions, exact solutions, the reference
solver, sampling, metrics, and plotting. Imports **no** ML framework.
"""

from __future__ import annotations
