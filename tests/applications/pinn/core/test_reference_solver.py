# SPDX-License-Identifier: Apache-2.0
"""Tests for the Godunov reference solver."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from applications.pinn.core import exact
from applications.pinn.core.reference_solver import godunov


def _burgers_flux(u: np.ndarray) -> np.ndarray:
    return 0.5 * u**2


def _burgers_flux_prime(u: np.ndarray) -> np.ndarray:
    return u


def _centers(a: float, b: float, nx: int) -> np.ndarray:
    edges = np.linspace(a, b, nx + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def test_converges_to_burgers_riemann_shock() -> None:
    """L1 error to the exact shock decreases under grid refinement."""
    a, b, t = -2.0, 3.0, 1.5
    u_l, u_r = 1.0, 0.0
    errors = []
    for nx in (200, 400, 800):
        x = _centers(a, b, nx)
        u0 = exact.burgers_riemann(x, 0.0, u_l, u_r)
        field = godunov(
            u0, x, [t], flux=_burgers_flux, flux_prime=_burgers_flux_prime, sonic=0.0
        )
        ref = exact.burgers_riemann(x, t, u_l, u_r)
        dx = (b - a) / nx
        errors.append(float(np.abs(field[0] - ref).sum() * dx))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 2e-2


def test_shock_speed_is_correct() -> None:
    """The recovered shock sits at s*t within one cell."""
    a, b, t, nx = -2.0, 3.0, 1.5, 800
    u_l, u_r = 1.0, 0.0
    s = 0.5 * (u_l + u_r)
    x = _centers(a, b, nx)
    u0 = exact.burgers_riemann(x, 0.0, u_l, u_r)
    field = godunov(
        u0, x, [t], flux=_burgers_flux, flux_prime=_burgers_flux_prime, sonic=0.0
    )[0]
    # midpoint crossing of (u_l+u_r)/2
    crossing = x[np.argmin(np.abs(field - 0.5 * (u_l + u_r)))]
    assert abs(crossing - s * t) < 2 * (b - a) / nx


def test_total_variation_non_increasing() -> None:
    """TV(t) is non-increasing (TVD), here constant for monotone data."""
    a, b, nx = -2.0, 3.0, 400
    x = _centers(a, b, nx)
    u0 = exact.burgers_riemann(x, 0.0, 1.0, 0.0)
    ts = [0.0, 0.5, 1.0, 1.5]
    field = godunov(
        u0, x, ts, flux=_burgers_flux, flux_prime=_burgers_flux_prime, sonic=0.0
    )
    tvs = [float(np.abs(np.diff(field[i])).sum()) for i in range(len(ts))]
    for earlier, later in pairwise(tvs):
        assert later <= earlier + 1e-9


def test_converges_to_lwr_riemann() -> None:
    """Godunov with the concave Greenshields flux matches the LWR shock."""
    a, b, t, nx = -2.0, 2.0, 1.0, 800
    rho_l, rho_r = 0.2, 0.8
    v_max, rho_max = 1.0, 1.0

    def flux(r: np.ndarray) -> np.ndarray:
        return exact.greenshields_flux(r, v_max, rho_max)

    def flux_prime(r: np.ndarray) -> np.ndarray:
        return exact.greenshields_flux_prime(r, v_max, rho_max)

    x = _centers(a, b, nx)
    rho0 = exact.lwr_riemann(x, 0.0, rho_l, rho_r, v_max=v_max, rho_max=rho_max)
    field = godunov(
        rho0, x, [t], flux=flux, flux_prime=flux_prime, sonic=rho_max / 2.0
    )[0]
    ref = exact.lwr_riemann(x, t, rho_l, rho_r, v_max=v_max, rho_max=rho_max)
    dx = (b - a) / nx
    assert float(np.abs(field - ref).sum() * dx) < 2e-2
