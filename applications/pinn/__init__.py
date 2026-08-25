# SPDX-License-Identifier: Apache-2.0
"""Structure-Preserving PINNs (Paper 1).

Hard monotonicity as a PDE admissibility prior. This package must import
without pulling in any ML backend: access backend code explicitly via
``applications.pinn.models.jax`` / ``applications.pinn.models.torch``.
"""

from __future__ import annotations
