# SPDX-License-Identifier: Apache-2.0
r"""NGSIM I-80 preprocessing: trajectories -> Edie density field -> monotone window.

Offline, NumPy-only. Reads raw NGSIM vehicle trajectories (10 Hz), aggregates a
dense density field via Edie's generalized definitions, picks the monotone
single-wave window by an objective scan, calibrates a fundamental diagram, and
writes the small derived ``.npz`` the ``ngsim_wave`` problem consumes.

Example::

    uv run python -m applications.pinn.data.ngsim --raw data/raw/i80.csv \
        --out applications/pinn/data/ngsim-i80-wave.npz
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def edie_fields(
    vehicle_id: Array,
    t: Array,
    x: Array,
    *,
    x_edges: Array,
    t_edges: Array,
) -> tuple[Array, Array]:
    """Aggregate trajectories into Edie density and flow fields.

    For each space-time cell ``A`` of area ``|A|``, Edie's generalized definitions
    give total travelled distance ``d(A)`` and total travel time ``t(A)``, then
    ``rho = t(A)/|A|`` and ``q = d(A)/|A|``. Each consecutive trajectory sample
    pair contributes its distance and time increment to the cell of its midpoint.
    Each cell is divided by its own area (handling non-uniform grids correctly).

    :param vehicle_id: Per-sample vehicle id (used to break increments between
        vehicles).
    :param t: Per-sample time.
    :param x: Per-sample position along the road.
    :param x_edges: Monotone spatial cell edges (length ``nx + 1``).
    :param t_edges: Monotone temporal cell edges (length ``nt + 1``).
    :returns: ``(rho, q)`` fields of shape ``(nt, nx)``.
    """
    order = np.lexsort((t, vehicle_id))
    vid, ts, xs = vehicle_id[order], t[order], x[order]
    same = vid[1:] == vid[:-1]
    dx_inc = np.where(same, xs[1:] - xs[:-1], 0.0)
    dt_inc = np.where(same, ts[1:] - ts[:-1], 0.0)
    mid_x = 0.5 * (xs[1:] + xs[:-1])
    mid_t = 0.5 * (ts[1:] + ts[:-1])
    xi = np.digitize(mid_x, x_edges) - 1
    ti = np.digitize(mid_t, t_edges) - 1
    nt, nx = len(t_edges) - 1, len(x_edges) - 1
    inside = (xi >= 0) & (xi < nx) & (ti >= 0) & (ti < nt) & same
    cell = ti[inside] * nx + xi[inside]
    dist = np.bincount(cell, weights=dx_inc[inside], minlength=nt * nx)
    time = np.bincount(cell, weights=dt_inc[inside], minlength=nt * nx)
    # Per-cell area (handles non-uniform edges correctly).
    cell_area = np.outer(np.diff(t_edges), np.diff(x_edges)).ravel()
    rho = (time / cell_area).reshape(nt, nx)
    q = (dist / cell_area).reshape(nt, nx)
    return rho, q


def calibrate_greenshields(rho: Array, q: Array) -> dict[str, float]:
    """Fit a Greenshields fundamental diagram to (rho, q) cell samples.

    Greenshields is ``q = v_max * rho * (1 - rho/rho_max)``, so speed
    ``v = q/rho = v_max - (v_max/rho_max) * rho`` is linear in density. We fit that
    line (intercept ``v_max``, slope ``-v_max/rho_max``). Greenshields is chosen
    over a triangular FD for the PDE residual because its characteristic speed
    ``f'(rho) = v_max(1 - 2 rho/rho_max)`` is smooth (triangular's is
    discontinuous at the critical density).

    :param rho: Per-cell density samples.
    :param q: Per-cell flow samples.
    :returns: ``{"v_max": ..., "rho_max": ...}``.
    :raises ValueError: If fewer than two positive-density samples are present.
    """
    mask = rho > 1e-9
    if int(mask.sum()) < 2:
        raise ValueError("need >= 2 positive-density cells to calibrate")
    r = rho[mask]
    v = q[mask] / r
    coeffs = np.polyfit(r, v, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])
    v_max = intercept
    rho_max = float(-v_max / slope) if slope < 0 else float(r.max() * 2.0)
    return {"v_max": v_max, "rho_max": rho_max}
