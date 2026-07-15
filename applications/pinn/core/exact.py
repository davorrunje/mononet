# SPDX-License-Identifier: Apache-2.0
"""Closed-form entropy solutions for scalar conservation laws.

These are the exact ground truth used to validate the reference solver and to
score PINNs on the forward mechanism tier. All functions are elementwise over
NumPy arrays and handle ``t == 0`` (the initial step) without division issues.

Conventions: the Riemann discontinuity sits at ``x0`` (default ``0``); ``x`` and
``t`` broadcast against each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

Array = npt.NDArray[np.floating]


def _riemann_step(x: Array, x0: float, left: float, right: float) -> Array:
    """Return the initial Riemann step: ``left`` for ``x < x0`` else ``right``."""
    return np.where(x < x0, left, right)


def burgers_riemann(
    x: Array,
    t: Array | float,
    u_l: float,
    u_r: float,
    *,
    x0: float = 0.0,
) -> Array:
    """Entropy solution of Burgers' equation for Riemann data.

    Flux ``f(u) = u^2 / 2`` (convex). For ``u_l > u_r`` a shock travels at the
    Rankine-Hugoniot speed ``s = (u_l + u_r) / 2``; for ``u_l < u_r`` a
    rarefaction fan connects the states; equal states are constant.

    :param x: Spatial coordinates.
    :param t: Time(s), broadcast against ``x``; ``t == 0`` gives the initial step.
    :param u_l: Left state.
    :param u_r: Right state.
    :param x0: Location of the initial discontinuity.
    :returns: ``u(x, t)`` on the entropy solution.
    """
    x = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    t_safe = np.where(t_arr > 0.0, t_arr, 1.0)
    xi = (x - x0) / t_safe
    step = _riemann_step(x, x0, u_l, u_r)
    if u_l > u_r:  # shock
        s = 0.5 * (u_l + u_r)
        moving = np.where(xi < s, u_l, u_r)
    elif u_l < u_r:  # rarefaction: u = xi clamped to [u_l, u_r]
        moving = np.clip(xi, u_l, u_r)
    else:
        moving = np.full_like(x, u_l)
    return np.where(t_arr > 0.0, moving, step)


def advection(
    x: Array,
    t: Array | float,
    a: float,
    u0: Callable[[Array], Array],
) -> Array:
    """Exact solution of linear advection ``u_t + a u_x = 0``.

    The initial profile is transported unchanged: ``u(x, t) = u0(x - a t)``.

    :param x: Spatial coordinates.
    :param t: Time(s), broadcast against ``x``.
    :param a: Constant advection speed.
    :param u0: Initial-condition callable.
    :returns: ``u(x, t)``.
    """
    x = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    return u0(x - a * t_arr)


def greenshields_flux(rho: Array, v_max: float, rho_max: float) -> Array:
    """Greenshields LWR flux ``Q(rho) = v_max * rho * (1 - rho / rho_max)``."""
    return v_max * rho * (1.0 - rho / rho_max)


def greenshields_flux_prime(rho: Array, v_max: float, rho_max: float) -> Array:
    """Characteristic speed ``Q'(rho) = v_max * (1 - 2 rho / rho_max)``."""
    return v_max * (1.0 - 2.0 * rho / rho_max)


def hindered_settling_flux(c: Array, v0: float, c_max: float, n: float = 2.0) -> Array:
    """Kynch batch-settling flux ``f_bk(c) = v0 * c * (1 - c/c_max)**n``.

    The hindered-settling (Michaels-Bolger / Richardson-Zaki) flux density for
    batch sedimentation: a scalar conservation law ``c_t + f_bk(c)_z = 0`` whose
    entropy solution is a monotone concentration profile with a descending
    settling interface. Backend-polymorphic (only ``*``, ``-``, ``/``, ``**``).

    :param c: Solids concentration (assumed in ``[0, c_max]``).
    :param v0: Reference (zero-concentration) settling velocity.
    :param c_max: Maximum (jam) concentration.
    :param n: Hindrance exponent (``2`` keeps the flux polynomial and smooth).
    :returns: The batch flux density ``f_bk(c)``.
    """
    return v0 * c * (1.0 - c / c_max) ** n


def hindered_settling_flux_prime(
    c: Array, v0: float, c_max: float, n: float = 2.0
) -> Array:
    """Characteristic speed ``f_bk'(c) = v0 (1 - c/c_max)**(n-1) (1 - (n+1) c/c_max)``.

    Analytic derivative of :func:`hindered_settling_flux`; backend-polymorphic.

    :param c: Solids concentration (assumed in ``[0, c_max]``).
    :param v0: Reference settling velocity.
    :param c_max: Maximum (jam) concentration.
    :param n: Hindrance exponent.
    :returns: The characteristic speed ``f_bk'(c)``.
    """
    return v0 * (1.0 - c / c_max) ** (n - 1.0) * (1.0 - (n + 1.0) * c / c_max)


def lwr_riemann(
    x: Array,
    t: Array | float,
    rho_l: float,
    rho_r: float,
    *,
    v_max: float = 1.0,
    rho_max: float = 1.0,
    x0: float = 0.0,
) -> Array:
    """Entropy solution of the LWR traffic model (Greenshields, concave flux).

    For the concave flux, ``rho_l < rho_r`` gives a shock (a forming queue)
    travelling at ``s = (Q(rho_r) - Q(rho_l)) / (rho_r - rho_l)``; ``rho_l >
    rho_r`` gives a rarefaction (a dissolving queue). Both profiles are monotone
    in ``x``.

    :param x: Spatial coordinates.
    :param t: Time(s), broadcast against ``x``; ``t == 0`` gives the initial step.
    :param rho_l: Left density state.
    :param rho_r: Right density state.
    :param v_max: Free-flow speed.
    :param rho_max: Jam density.
    :param x0: Location of the initial discontinuity.
    :returns: ``rho(x, t)`` on the entropy solution.
    """
    x = np.asarray(x, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    t_safe = np.where(t_arr > 0.0, t_arr, 1.0)
    xi = (x - x0) / t_safe
    step = _riemann_step(x, x0, rho_l, rho_r)
    if rho_l < rho_r:  # shock
        q_l = greenshields_flux(np.asarray(rho_l), v_max, rho_max)
        q_r = greenshields_flux(np.asarray(rho_r), v_max, rho_max)
        s = float((q_r - q_l) / (rho_r - rho_l))
        moving = np.where(xi < s, rho_l, rho_r)
    elif rho_l > rho_r:  # rarefaction
        lam_l = float(greenshields_flux_prime(np.asarray(rho_l), v_max, rho_max))
        lam_r = float(greenshields_flux_prime(np.asarray(rho_r), v_max, rho_max))
        fan = 0.5 * rho_max * (1.0 - xi / v_max)  # invert Q'(rho) = xi
        moving = np.where(xi <= lam_l, rho_l, np.where(xi >= lam_r, rho_r, fan))
    else:
        moving = np.full_like(x, rho_l)
    return np.where(t_arr > 0.0, moving, step)


def breaking_time(du0_dx_min: float) -> float:
    """Shock breaking time ``t_b = -1 / min(u0')`` for Burgers.

    :param du0_dx_min: The minimum of the initial-derivative ``u0'`` (negative
        for shock-forming data).
    :returns: The finite breaking time.
    :raises ValueError: If ``du0_dx_min >= 0`` (no shock forms).
    """
    if du0_dx_min >= 0.0:
        raise ValueError("no shock forms when min(u0') >= 0")
    return -1.0 / du0_dx_min


def burgers_characteristic(
    x: Array,
    t: float,
    u0: Callable[[Array], Array],
    *,
    iters: int = 500,
    tol: float = 1e-13,
) -> Array:
    """Pre-shock Burgers solution via the method of characteristics.

    Solves the implicit relation ``u = u0(x - u t)`` by fixed-point iteration,
    which converges for ``t < t_b`` (before any shock forms). Use the reference
    solver for ``t >= t_b``.

    :param x: Spatial coordinates.
    :param t: Time (scalar), must be strictly before the breaking time.
    :param u0: Initial-condition callable.
    :param iters: Maximum fixed-point iterations.
    :param tol: Convergence tolerance on successive iterates.
    :returns: ``u(x, t)``.
    """
    x = np.asarray(x, dtype=float)
    u = u0(x)
    for _ in range(iters):
        u_next = u0(x - u * t)
        if float(np.max(np.abs(u_next - u))) < tol:
            return u_next
        u = u_next
    return u
