# SPDX-License-Identifier: Apache-2.0
"""Real-data traffic problem backed by an NGSIM-derived density field.

``ngsim_wave`` loads the committed ``.npz`` (produced by
``applications.pinn.data.ngsim``): a dense Edie density field on a monotone
single-wave window, plus a calibrated Greenshields fundamental diagram. The
``ground_truth`` is bilinear interpolation of that field (the best-estimate
reference), not a closed-form exact solution.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from applications.pinn.core import exact
from applications.pinn.core.admissibility import AdmissibilitySpec
from applications.pinn.core.problems.base import register

Array = npt.NDArray[np.floating]

_DEFAULT_NPZ = str(Path(__file__).resolve().parents[2] / "data" / "ngsim-i80-wave.npz")


@register("ngsim_wave")
class NgsimWave:
    """LWR density reconstruction on a real NGSIM I-80 stop-and-go wave window."""

    key: ClassVar[str] = "ngsim_wave"

    def __init__(self, npz_path: str = _DEFAULT_NPZ) -> None:
        """Load the derived field + FD params from ``npz_path``.

        :param npz_path: Path to the ``.npz`` written by the ngsim preprocessor.
        """
        d = np.load(npz_path, allow_pickle=True)
        self._x = d["x"].astype(np.float64)
        self._t = d["t"].astype(np.float64)
        self._rho = d["rho"].astype(np.float64)
        self.v_max = float(d["v_max"])
        self.rho_max = float(d["rho_max"])
        self._sign_x = int(d["sign_x"])
        self.monotonicity_defect = float(d["monotonicity_defect"])
        self._interp = RegularGridInterpolator(
            (self._t, self._x), self._rho, bounds_error=False, fill_value=None
        )

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((x_min, x_max), (t_min, t_max))`` from the loaded grid."""
        return (
            (float(self._x[0]), float(self._x[-1])),
            (float(self._t[0]), float(self._t[-1])),
        )

    def admissibility(self) -> AdmissibilitySpec:
        """Monotone in ``x`` in the window's density-gradient direction."""
        return AdmissibilitySpec(mask=(self._sign_x, 0))

    def flux(self, u: Array) -> Array:
        """Calibrated Greenshields flux (backend-polymorphic)."""
        return exact.greenshields_flux(u, self.v_max, self.rho_max)

    def flux_prime(self, u: Array) -> Array:
        """Calibrated Greenshields characteristic speed (backend-polymorphic)."""
        return exact.greenshields_flux_prime(u, self.v_max, self.rho_max)

    def initial(self, x: Array) -> Array:
        """Density at the window's initial time (interpolated)."""
        t0 = np.full_like(np.asarray(x, dtype=np.float64), self._t[0])
        return self.ground_truth(np.asarray(x, dtype=np.float64), t0)

    def ground_truth(self, x: Array, t: Array) -> Array:
        """Bilinear interpolation of the Edie reference field at ``(x, t)``."""
        pts = np.column_stack(
            [
                np.asarray(t, dtype=np.float64).ravel(),
                np.asarray(x, dtype=np.float64).ravel(),
            ]
        )
        return np.asarray(self._interp(pts), dtype=np.float64)
