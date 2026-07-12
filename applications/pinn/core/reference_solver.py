# SPDX-License-Identifier: Apache-2.0
"""A TVD finite-volume (Godunov) reference solver for scalar conservation laws.

This is the numerical ground truth: a first-order Godunov scheme that converges
to the entropy solution and is total-variation-diminishing by construction. It
handles both convex (Burgers) and concave (LWR) fluxes through the exact Godunov
interface flux, which needs only the flux and its single sonic point ``u*``
(where ``f'(u*) = 0``). Outflow (zero-gradient) boundaries; CFL-limited stepping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

Array = npt.NDArray[np.floating]


def godunov_flux(
    u_left: Array,
    u_right: Array,
    flux: Callable[[Array], Array],
    sonic: float,
) -> Array:
    """Exact Godunov numerical flux at cell interfaces.

    For ``u_left <= u_right`` the flux is the minimum of ``f`` over
    ``[u_left, u_right]``; otherwise the maximum over ``[u_right, u_left]``. With
    a single sonic point ``u*`` these extrema are attained at the endpoints or at
    ``u*`` when it lies inside the interval — exact for convex or concave ``f``.

    :param u_left: Left interface states.
    :param u_right: Right interface states.
    :param flux: Flux function ``f``.
    :param sonic: The point ``u*`` where ``f'(u*) = 0``.
    :returns: Godunov flux per interface.
    """
    f_l = flux(u_left)
    f_r = flux(u_right)
    f_s = flux(np.full_like(u_left, sonic))
    lo = np.minimum(u_left, u_right)
    hi = np.maximum(u_left, u_right)
    sonic_inside = (lo <= sonic) & (sonic <= hi)
    minimum = np.minimum(f_l, f_r)
    maximum = np.maximum(f_l, f_r)
    min_flux = np.where(sonic_inside, np.minimum(minimum, f_s), minimum)
    max_flux = np.where(sonic_inside, np.maximum(maximum, f_s), maximum)
    return np.where(u_left <= u_right, min_flux, max_flux)


def godunov(
    u0_values: Array,
    x_centers: Array,
    t_eval: Sequence[float],
    *,
    flux: Callable[[Array], Array],
    flux_prime: Callable[[Array], Array],
    sonic: float,
    cfl: float = 0.4,
) -> Array:
    """Evolve initial data with the Godunov scheme, snapshotting at ``t_eval``.

    :param u0_values: Initial cell-centre values.
    :param x_centers: Uniform cell-centre coordinates.
    :param t_eval: Non-negative, increasing times at which to snapshot.
    :param flux: Flux ``f``.
    :param flux_prime: Flux derivative ``f'`` (for the CFL condition).
    :param sonic: Sonic point ``u*`` (``f'(u*) = 0``).
    :param cfl: CFL number in ``(0, 1)``.
    :returns: Array of shape ``(len(t_eval), len(x_centers))``.
    """
    dx = float(x_centers[1] - x_centers[0])
    u = np.array(u0_values, dtype=float)
    snapshots: list[Array] = []
    current = 0.0
    for target in t_eval:
        while current < target - 1e-15:
            speed = float(np.max(np.abs(flux_prime(u)))) + 1e-12
            dt = min(cfl * dx / speed, target - current)
            extended = np.concatenate([u[:1], u, u[-1:]])
            interface = godunov_flux(extended[:-1], extended[1:], flux, sonic)
            u = u - (dt / dx) * (interface[1:] - interface[:-1])
            current += dt
        snapshots.append(u.copy())
    return np.asarray(snapshots)


def interpolate(
    field: Array,
    x_values: Array,
    t_values: Array,
    x_query: Array,
    t_query: Array,
) -> Array:
    """Bilinearly interpolate a reference ``field`` at scattered query points.

    :param field: Field of shape ``(len(t_values), len(x_values))``.
    :param x_values: Ascending spatial grid axis.
    :param t_values: Ascending temporal grid axis.
    :param x_query: Query x-coordinates.
    :param t_query: Query t-coordinates (same shape as ``x_query``).
    :returns: Interpolated values, shape of ``x_query``.
    """
    xq = np.asarray(x_query, dtype=float)
    tq = np.asarray(t_query, dtype=float)
    ix = np.clip(np.searchsorted(x_values, xq) - 1, 0, len(x_values) - 2)
    it = np.clip(np.searchsorted(t_values, tq) - 1, 0, len(t_values) - 2)
    x0, x1 = x_values[ix], x_values[ix + 1]
    t0, t1 = t_values[it], t_values[it + 1]
    wx = np.clip((xq - x0) / (x1 - x0), 0.0, 1.0)
    wt = np.clip((tq - t0) / (t1 - t0), 0.0, 1.0)
    top = field[it, ix] * (1 - wx) + field[it, ix + 1] * wx
    bot = field[it + 1, ix] * (1 - wx) + field[it + 1, ix + 1] * wx
    return top * (1 - wt) + bot * wt
