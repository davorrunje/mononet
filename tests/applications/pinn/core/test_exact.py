# SPDX-License-Identifier: Apache-2.0
"""Tests for closed-form entropy solutions."""

from __future__ import annotations

import numpy as np

from applications.pinn.core import exact


def test_burgers_shock_speed_and_monotonicity() -> None:
    """u_l>u_r gives a shock at s=(u_l+u_r)/2, monotone decreasing in x."""
    u_l, u_r, t = 1.0, 0.0, 2.0
    s = 0.5 * (u_l + u_r)
    x = np.linspace(-2.0, 3.0, 401)
    u = exact.burgers_riemann(x, t, u_l, u_r)
    # left of the shock -> u_l, right -> u_r
    assert np.all(u[x < s * t - 0.05] == u_l)
    assert np.all(u[x > s * t + 0.05] == u_r)
    # monotone non-increasing
    assert np.all(np.diff(u) <= 1e-12)


def test_burgers_rarefaction_is_self_similar() -> None:
    """u_l<u_r gives a rarefaction: u = x/t clamped, monotone increasing."""
    u_l, u_r, t = 0.0, 1.0, 1.5
    x = np.linspace(-1.0, 3.0, 401)
    u = exact.burgers_riemann(x, t, u_l, u_r)
    xi = x / t
    inside = (xi > u_l) & (xi < u_r)
    assert np.allclose(u[inside], xi[inside])
    assert np.all(np.diff(u) >= -1e-12)


def test_burgers_initial_step_at_t0() -> None:
    """At t=0 the solution is the Riemann step."""
    x = np.array([-1.0, -0.1, 0.1, 1.0])
    u = exact.burgers_riemann(x, 0.0, 2.0, 0.5)
    assert np.array_equal(u, np.array([2.0, 2.0, 0.5, 0.5]))


def test_advection_translates_profile() -> None:
    """Linear advection transports the initial profile at speed a."""
    a, t = 0.75, 2.0

    def u0(x: np.ndarray) -> np.ndarray:
        out: np.ndarray = np.exp(-(x**2))
        return out

    x = np.linspace(-3.0, 5.0, 200)
    u = exact.advection(x, t, a, u0)
    assert np.allclose(u, u0(x - a * t))


def test_lwr_shock_speed_rankine_hugoniot() -> None:
    """LWR shock (rho_l<rho_r) moves at the Rankine-Hugoniot speed."""
    rho_l, rho_r, t = 0.2, 0.8, 1.0
    v_max, rho_max = 1.0, 1.0
    q_l = exact.greenshields_flux(np.asarray(rho_l), v_max, rho_max)
    q_r = exact.greenshields_flux(np.asarray(rho_r), v_max, rho_max)
    s = float((q_r - q_l) / (rho_r - rho_l))
    x = np.linspace(-2.0, 2.0, 401)
    rho = exact.lwr_riemann(x, t, rho_l, rho_r, v_max=v_max, rho_max=rho_max)
    assert np.all(rho[x < s * t - 0.05] == rho_l)
    assert np.all(rho[x > s * t + 0.05] == rho_r)
    # density increasing in x (forming queue)
    assert np.all(np.diff(rho) >= -1e-12)


def test_lwr_rarefaction_monotone_decreasing() -> None:
    """LWR rarefaction (rho_l>rho_r) is a monotone-decreasing fan."""
    rho_l, rho_r, t = 0.8, 0.2, 1.0
    x = np.linspace(-2.0, 2.0, 401)
    rho = exact.lwr_riemann(x, t, rho_l, rho_r)
    assert np.all(np.diff(rho) <= 1e-12)
    assert np.isclose(rho[0], rho_l)
    assert np.isclose(rho[-1], rho_r)


def test_breaking_time() -> None:
    """t_b = -1/min(u0'); raises when no shock forms."""
    assert np.isclose(exact.breaking_time(-2.0), 0.5)
    with np.testing.assert_raises(ValueError):
        exact.breaking_time(1.0)


def test_characteristic_satisfies_implicit_relation() -> None:
    """Pre-shock solution satisfies u = u0(x - u t) to tolerance."""

    def u0(x: np.ndarray) -> np.ndarray:
        out: np.ndarray = -np.arctan(x)  # decreasing; min slope -1 -> t_b=1
        return out

    t = 0.5  # < t_b = 1
    x = np.linspace(-4.0, 4.0, 200)
    u = exact.burgers_characteristic(x, t, u0)
    assert np.max(np.abs(u - u0(x - u * t))) < 1e-10


def test_hindered_settling_flux_shape_and_derivative() -> None:
    """Kynch batch flux vanishes at c=0 and c=c_max; prime matches finite diff."""
    v0, c_max = 1.0e-3, 10.0
    c = np.linspace(0.0, c_max, 200)
    f = exact.hindered_settling_flux(c, v0, c_max)
    assert abs(float(f[0])) < 1e-12  # f(0) = 0
    assert abs(float(f[-1])) < 1e-12  # f(c_max) = 0
    assert float(f.max()) > 0.0  # positive settling flux in between
    # analytic derivative agrees with a central finite difference
    fp = exact.hindered_settling_flux_prime(c, v0, c_max)
    fd = np.gradient(f, c)
    inner = slice(2, -2)
    assert np.max(np.abs(fp[inner] - fd[inner])) < 1e-4
