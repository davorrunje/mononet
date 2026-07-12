# SPDX-License-Identifier: Apache-2.0
"""Invariant: the pure-NumPy core imports no ML framework."""

from __future__ import annotations

import subprocess
import sys


def test_core_imports_no_backend() -> None:
    """Importing the whole core (incl. the problem registry) pulls in no backend.

    Run in a fresh interpreter so the check is independent of test order.
    """
    code = (
        "import sys\n"
        "import applications.pinn.core.problems\n"
        "import applications.pinn.core.metrics\n"
        "import applications.pinn.core.sampling\n"
        "import applications.pinn.core.reference_solver\n"
        "assert 'torch' not in sys.modules, 'torch imported by core'\n"
        "assert 'jax' not in sys.modules, 'jax imported by core'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
