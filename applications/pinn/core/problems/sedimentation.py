# SPDX-License-Identifier: Apache-2.0
"""Batch sedimentation (Kynch) problem backed by a measured concentration field.

``sediment_batch`` is the real-data counterpart for the sedimentation regime: the
solids concentration during batch settling obeys a scalar conservation law
``C_t + f_bk(C)_z = 0`` with the Kynch hindered-settling flux, and its entropy
solution is **monotone in height** (dense sediment below, clear liquid above, a
descending interface). It loads a measured ``C(z, t)`` field from a ``.npz``
(e.g. the De Clercq radiotracer profiles once obtained); ``ground_truth`` is the
best-estimate reference by interpolation, exactly like the traffic problem.

The flux is the backend-polymorphic Michaels-Bolger / Kynch form in
:mod:`applications.pinn.core.exact`; a compression (degenerate-diffusion) term is
a documented future extension and is not part of the hyperbolic residual here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import numpy.typing as npt

from applications.pinn.core import exact
from applications.pinn.core.admissibility import AdmissibilitySpec
from applications.pinn.core.problems.base import register

if TYPE_CHECKING:
    from typing import Any

Array = npt.NDArray[np.floating]

_DEFAULT_NPZ = str(Path(__file__).resolve().parents[2] / "data" / "declercq-batch.npz")


@register("sediment_batch")
class SedimentBatch:
    """Batch-settling solids-concentration reconstruction (Kynch conservation law)."""

    key: ClassVar[str] = "sediment_batch"

    def __init__(self, npz_path: str = _DEFAULT_NPZ) -> None:
        """Load the measured ``C(z, t)`` field and Kynch flux parameters.

        Expected ``.npz`` keys: ``x`` (height axis ``z``, ``(nx,)``), ``t``
        (``(nt,)``), ``rho`` (concentration field ``(nt, nx)``), ``v0`` (reference
        settling velocity), ``c_max`` (jam concentration), ``sign_x`` (admissible
        sign of ``dC/dz``), and optionally ``n`` (hindrance exponent, default 2).

        :param npz_path: Path to the ``.npz`` produced from the measured profiles.
        """
        from scipy.interpolate import RegularGridInterpolator

        d = np.load(npz_path, allow_pickle=True)
        self._z = d["x"].astype(np.float64)
        self._t = d["t"].astype(np.float64)
        self._c = d["rho"].astype(np.float64)
        self.v0 = float(d["v0"])
        self.c_max = float(d["c_max"])
        self.n = float(d["n"]) if "n" in d else 2.0
        self._sign_x = int(d["sign_x"])
        self._interp = RegularGridInterpolator(
            (self._t, self._z), self._c, bounds_error=False, fill_value=None
        )

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((z_min, z_max), (t_min, t_max))`` from the loaded grid."""
        return (
            (float(self._z[0]), float(self._z[-1])),
            (float(self._t[0]), float(self._t[-1])),
        )

    def admissibility(self) -> AdmissibilitySpec:
        """Monotone in height ``z`` in the settling direction (stored sign)."""
        return AdmissibilitySpec(mask=(self._sign_x, 0))

    def flux(self, u: Array) -> Array:
        """Kynch hindered-settling batch flux (backend-polymorphic)."""
        return exact.hindered_settling_flux(u, self.v0, self.c_max, self.n)

    def flux_prime(self, u: Array) -> Array:
        """Kynch batch-flux characteristic speed (backend-polymorphic)."""
        return exact.hindered_settling_flux_prime(u, self.v0, self.c_max, self.n)

    def initial(self, x: Array) -> Array:
        """Concentration at the initial time (interpolated from the field)."""
        t0 = np.full_like(np.asarray(x, dtype=np.float64), self._t[0])
        return self.ground_truth(np.asarray(x, dtype=np.float64), t0)

    def ground_truth(self, x: Array, t: Array) -> Array:
        """Bilinear interpolation of the measured ``C(z, t)`` field."""
        pts = np.column_stack(
            [
                np.asarray(t, dtype=np.float64).ravel(),
                np.asarray(x, dtype=np.float64).ravel(),
            ]
        )
        result: Any = self._interp(pts)
        return np.asarray(result, dtype=np.float64)
