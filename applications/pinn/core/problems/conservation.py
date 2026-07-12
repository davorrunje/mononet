# SPDX-License-Identifier: Apache-2.0
"""Scalar conservation-law problems: Burgers, linear advection, LWR traffic.

Each problem is registered and exposes the flux, the admissibility spec (the
monotone direction of the entropy solution in ``x``), the initial condition, and
a ground-truth evaluator. All ground truth is closed-form except the smooth Burgers
case, which forms a shock and uses the Godunov reference past the breaking time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from applications.pinn.core import exact, reference_solver
from applications.pinn.core.admissibility import AdmissibilitySpec
from applications.pinn.core.problems.base import register

Array = npt.NDArray[np.floating]


def _mono_sign(left: float, right: float) -> int:
    """Return the monotone direction in ``x`` of a left>right / left<right step."""
    return -1 if left > right else 1


@register("burgers_riemann")
@dataclass(frozen=True, slots=True)
class BurgersRiemann:
    """Burgers' equation with Riemann initial data (pure shock or rarefaction)."""

    key: ClassVar[str] = "burgers_riemann"
    u_l: float = 1.0
    u_r: float = 0.0
    x0: float = 0.0
    x_range: tuple[float, float] = (-2.0, 3.0)
    t_max: float = 1.5

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the space-time domain."""
        return (self.x_range, (0.0, self.t_max))

    @property
    def sonic(self) -> float:
        """Sonic point of the convex Burgers flux."""
        return 0.0

    def admissibility(self) -> AdmissibilitySpec:
        """Monotone in ``x`` in the direction set by the Riemann states."""
        return AdmissibilitySpec(mask=(_mono_sign(self.u_l, self.u_r), 0))

    def flux(self, u: Array) -> Array:
        """Burgers flux ``u^2 / 2`` (backend-polymorphic)."""
        return 0.5 * u**2

    def flux_prime(self, u: Array) -> Array:
        """Characteristic speed ``u`` (backend-polymorphic)."""
        return u

    def initial(self, x: Array) -> Array:
        """Return the initial Riemann step."""
        return exact.burgers_riemann(x, 0.0, self.u_l, self.u_r, x0=self.x0)

    def ground_truth(self, x: Array, t: Array) -> Array:
        """Exact entropy solution."""
        return exact.burgers_riemann(x, t, self.u_l, self.u_r, x0=self.x0)


@register("advection")
@dataclass(frozen=True, slots=True)
class LinearAdvection:
    """Linear advection of a monotone-decreasing front at constant speed ``a``."""

    key: ClassVar[str] = "advection"
    a: float = 1.0
    steepness: float = 3.0
    x0: float = 0.0
    x_range: tuple[float, float] = (-3.0, 4.0)
    t_max: float = 1.5

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the space-time domain."""
        return (self.x_range, (0.0, self.t_max))

    def admissibility(self) -> AdmissibilitySpec:
        """Return the spec: the transported front is non-increasing in ``x``."""
        return AdmissibilitySpec(mask=(-1, 0))

    def _u0(self, x: Array) -> Array:
        profile = 0.5 * (1.0 - np.tanh(self.steepness * (x - self.x0)))
        return np.asarray(profile, dtype=float)

    def flux(self, u: Array) -> Array:
        """Linear flux ``a * u`` (backend-polymorphic)."""
        result: Array = self.a * u
        return result

    def flux_prime(self, u: Array) -> Array:
        """Constant characteristic speed ``a`` (backend-polymorphic broadcast)."""
        result: Array = self.a + 0.0 * u
        return result

    def initial(self, x: Array) -> Array:
        """Monotone-decreasing initial front."""
        return self._u0(np.asarray(x, dtype=float))

    def ground_truth(self, x: Array, t: Array) -> Array:
        """Exact transported profile ``u0(x - a t)``."""
        return exact.advection(x, t, self.a, self._u0)


@register("lwr_riemann")
@dataclass(frozen=True, slots=True)
class LwrRiemann:
    """LWR traffic (Greenshields flux) with Riemann density data."""

    key: ClassVar[str] = "lwr_riemann"
    rho_l: float = 0.2
    rho_r: float = 0.8
    v_max: float = 1.0
    rho_max: float = 1.0
    x0: float = 0.0
    x_range: tuple[float, float] = (-2.0, 2.0)
    t_max: float = 1.5

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the space-time domain."""
        return (self.x_range, (0.0, self.t_max))

    @property
    def sonic(self) -> float:
        """Sonic point of the concave Greenshields flux."""
        return self.rho_max / 2.0

    def admissibility(self) -> AdmissibilitySpec:
        """Monotone in ``x`` in the direction set by the density states."""
        return AdmissibilitySpec(mask=(_mono_sign(self.rho_l, self.rho_r), 0))

    def flux(self, u: Array) -> Array:
        """Greenshields flux (backend-polymorphic)."""
        return exact.greenshields_flux(u, self.v_max, self.rho_max)

    def flux_prime(self, u: Array) -> Array:
        """Greenshields characteristic speed (backend-polymorphic)."""
        return exact.greenshields_flux_prime(u, self.v_max, self.rho_max)

    def initial(self, x: Array) -> Array:
        """Return the initial Riemann density step."""
        return exact.lwr_riemann(
            x,
            0.0,
            self.rho_l,
            self.rho_r,
            v_max=self.v_max,
            rho_max=self.rho_max,
            x0=self.x0,
        )

    def ground_truth(self, x: Array, t: Array) -> Array:
        """Exact entropy solution."""
        return exact.lwr_riemann(
            x,
            t,
            self.rho_l,
            self.rho_r,
            v_max=self.v_max,
            rho_max=self.rho_max,
            x0=self.x0,
        )


@register("burgers_smooth")
@dataclass(slots=True)
class BurgersSmoothShock:
    """Burgers with smooth monotone-decreasing data that steepens into a shock.

    Exact pre-breaking (method of characteristics); past the breaking time the
    ground truth is the Godunov reference, interpolated. The reference field is
    built once and cached.
    """

    key: ClassVar[str] = "burgers_smooth"
    steepness: float = 1.0
    x_range: tuple[float, float] = (-4.0, 4.0)
    t_max: float = 2.0
    ref_nx: int = 1000
    ref_nt: int = 400
    _cache: dict[str, Array] = field(default_factory=dict, compare=False, repr=False)

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the space-time domain."""
        return (self.x_range, (0.0, self.t_max))

    @property
    def sonic(self) -> float:
        """Sonic point of the convex Burgers flux."""
        return 0.0

    @property
    def breaking_time(self) -> float:
        """Breaking time ``t_b = -1 / min(u0')`` for the chosen initial data."""
        # u0 = -tanh(k x): min slope is -k at x = 0.
        return 1.0 / self.steepness

    def admissibility(self) -> AdmissibilitySpec:
        """Monotone non-increasing in ``x``."""
        return AdmissibilitySpec(mask=(-1, 0))

    def _u0(self, x: Array) -> Array:
        return -np.tanh(self.steepness * np.asarray(x, dtype=float))

    def flux(self, u: Array) -> Array:
        """Burgers flux ``u^2 / 2`` (backend-polymorphic)."""
        return 0.5 * u**2

    def flux_prime(self, u: Array) -> Array:
        """Characteristic speed ``u`` (backend-polymorphic)."""
        return u

    def initial(self, x: Array) -> Array:
        """Smooth monotone-decreasing initial profile."""
        return self._u0(x)

    def _reference(self) -> tuple[Array, Array, Array]:
        if "field" not in self._cache:
            (x_lo, x_hi), (_t0, t_hi) = self.domain
            edges = np.linspace(x_lo, x_hi, self.ref_nx + 1)
            xv = 0.5 * (edges[:-1] + edges[1:])
            tv = np.linspace(0.0, t_hi, self.ref_nt)
            fld = reference_solver.godunov(
                self._u0(xv),
                xv,
                list(tv),
                flux=self.flux,
                flux_prime=self.flux_prime,
                sonic=self.sonic,
            )
            self._cache["x"], self._cache["t"], self._cache["field"] = xv, tv, fld
        return self._cache["x"], self._cache["t"], self._cache["field"]

    def ground_truth(self, x: Array, t: Array) -> Array:
        """Characteristic solution pre-breaking; Godunov reference afterwards."""
        xv, tv, fld = self._reference()
        return reference_solver.interpolate(fld, xv, tv, x, t)
