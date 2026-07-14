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

import argparse
import csv
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from applications.pinn.core.admissibility import violation

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


def _defect(rho_win: Array, sign: int) -> float:
    """Compute residual wrong-sign variation normalized by total variation."""
    v = sum(violation(rho_win[i], axis=0, sign=sign) for i in range(rho_win.shape[0]))
    tv = float(np.abs(np.diff(rho_win, axis=1)).sum())
    return v / tv if tv > 0 else 1.0


def window_scan(
    rho: Array,
    x: Array,
    t: Array,
    *,
    tau: float = 0.05,
    min_nx: int = 8,
    min_nt: int = 8,
) -> dict[str, object]:
    """Pick the largest space-time window whose density is nearly monotone in x.

    Real corridors carry several simultaneous waves and are globally non-monotone
    in x. We search sub-windows, score each by its ``monotonicity_defect`` (the
    reference field's own residual wrong-sign variation, from
    :func:`~applications.pinn.core.admissibility.violation`, normalized by total
    variation), and return the largest window under ``tau``.

    :param rho: Dense density field ``(nt, nx)``.
    :param x: Spatial axis ``(nx,)``.
    :param t: Temporal axis ``(nt,)``.
    :param tau: Maximum acceptable monotonicity defect (0 = perfectly monotone).
    :param min_nx: Minimum window width in cells.
    :param min_nt: Minimum window height in cells.
    :returns: ``{"xi": (lo, hi), "ti": (lo, hi), "sign_x": +/-1, "defect": float}``.
    :raises ValueError: If no window meets ``tau`` (caller should widen or pivot).
    """
    nt, nx = rho.shape
    best: dict[str, object] | None = None
    best_area = 0
    for x_lo in range(0, nx - min_nx + 1, 2):
        for x_hi in range(x_lo + min_nx, nx + 1, 2):
            for t_lo in range(0, nt - min_nt + 1, 2):
                for t_hi in range(t_lo + min_nt, nt + 1, 2):
                    win = rho[t_lo:t_hi, x_lo:x_hi]
                    sign = -1 if win[:, -1].mean() < win[:, 0].mean() else 1
                    defect = _defect(win, sign)
                    if defect < tau:
                        area = (x_hi - x_lo) * (t_hi - t_lo)
                        if area > best_area:
                            best_area = area
                            best = {
                                "xi": (x_lo, x_hi),
                                "ti": (t_lo, t_hi),
                                "sign_x": sign,
                                "defect": defect,
                            }
    if best is None:
        raise ValueError(f"no window with defect < {tau}; widen tau or use xi=x-ct")
    return best


def _load_raw(raw_csv: Path, lane: int) -> tuple[Array, Array, Array]:
    """Load (vehicle_id, t seconds, x metres) for one lane from an NGSIM CSV."""
    vids, ts, xs = [], [], []
    with Path(raw_csv).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(float(row["Lane_ID"])) != lane:
                continue
            vids.append(float(row["Vehicle_ID"]))
            ts.append(float(row["Global_Time"]) / 1000.0)  # ms -> s
            xs.append(float(row["Local_Y"]))  # already metres in re-metricated NGSIM
    return (
        np.asarray(vids, dtype=np.float64),
        np.asarray(ts, dtype=np.float64),
        np.asarray(xs, dtype=np.float64),
    )


def build_dataset(
    raw_csv: str | Path,
    out_npz: str | Path,
    *,
    lane: int = 1,
    dx: float = 20.0,
    dt: float = 5.0,
    tau: float = 0.05,
) -> dict[str, object]:
    """Build the derived ``.npz`` from a raw NGSIM CSV.

    Pipeline: load one lane -> Edie-aggregate the full section -> scan for the
    monotone window -> crop -> calibrate Greenshields on the window -> write npz.

    :param raw_csv: Path to the raw NGSIM trajectory CSV.
    :param out_npz: Output path for the derived artifact.
    :param lane: Lane id to select.
    :param dx: Spatial cell size (metres).
    :param dt: Temporal cell size (seconds).
    :param tau: Monotonicity-defect threshold for the window scan.
    :returns: A summary dict (window, defect, FD params, grid shape).
    """
    vid, t, x = _load_raw(Path(raw_csv), lane)
    x_edges = np.arange(x.min(), x.max() + dx, dx)
    t_edges = np.arange(t.min(), t.max() + dt, dt)
    rho, q = edie_fields(vid, t, x, x_edges=x_edges, t_edges=t_edges)
    x_c = 0.5 * (x_edges[:-1] + x_edges[1:])
    t_c = 0.5 * (t_edges[:-1] + t_edges[1:])
    win = window_scan(rho, x_c, t_c, tau=tau)
    xi = cast("tuple[int, int]", win["xi"])
    ti = cast("tuple[int, int]", win["ti"])
    x_lo, x_hi = xi
    t_lo, t_hi = ti
    rho_w = rho[t_lo:t_hi, x_lo:x_hi]
    q_w = q[t_lo:t_hi, x_lo:x_hi]
    x_w = x_c[x_lo:x_hi] - x_c[x_lo]  # re-origin to 0
    t_w = t_c[t_lo:t_hi] - t_c[t_lo]
    fd = calibrate_greenshields(rho_w, q_w)
    provenance = (
        f"NGSIM I-80, lane={lane}, dx={dx}m, dt={dt}s, "
        f"window x[{x_lo}:{x_hi}] t[{t_lo}:{t_hi}], tau={tau}"
    )
    np.savez(
        out_npz,
        x=x_w,
        t=t_w,
        rho=rho_w,
        q=q_w,
        v_max=fd["v_max"],
        rho_max=fd["rho_max"],
        sign_x=int(win["sign_x"]),  # type: ignore[call-overload]
        monotonicity_defect=float(win["defect"]),  # type: ignore[arg-type]
        provenance=provenance,
    )
    return {
        "nx": rho_w.shape[1],
        "nt": rho_w.shape[0],
        "window": win,
        "fd": fd,
        "out": str(out_npz),
    }


def main() -> None:
    """CLI: build the derived NGSIM ``.npz`` from a raw trajectory CSV."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="applications/pinn/data/ngsim-i80-wave.npz")
    p.add_argument("--lane", type=int, default=1)
    p.add_argument("--dx", type=float, default=20.0)
    p.add_argument("--dt", type=float, default=5.0)
    p.add_argument("--tau", type=float, default=0.05)
    args = p.parse_args()
    summary = build_dataset(
        args.raw, args.out, lane=args.lane, dx=args.dx, dt=args.dt, tau=args.tau
    )
    print(f"== wrote {args.out}: {summary} ==", flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
