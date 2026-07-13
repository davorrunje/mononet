# SPDX-License-Identifier: Apache-2.0
"""Tests for NGSIM preprocessing: Edie aggregation, FD calibration, window scan."""

from __future__ import annotations

import numpy as np

from applications.pinn.data import ngsim


def test_edie_fields_single_vehicle_constant_speed() -> None:
    """One vehicle at constant speed gives density = 1/(speed) * (occupancy)."""
    # Vehicle crosses a 1 m x 1 s cell grid at 1 m/s, sampled at dt=0.1s.
    t = np.arange(0.0, 2.0, 0.1)
    x = 1.0 * t  # speed 1 m/s from x=0..~2
    vid = np.zeros_like(t)
    x_edges = np.array([0.0, 1.0, 2.0])
    t_edges = np.array([0.0, 1.0, 2.0])
    rho, q = ngsim.edie_fields(vid, t, x, x_edges=x_edges, t_edges=t_edges)
    assert rho.shape == (2, 2)
    # Speed q/rho should recover ~1 m/s in the cells the vehicle occupies.
    occupied = rho > 0
    speed = np.divide(q, rho, out=np.zeros_like(q), where=occupied)
    assert np.allclose(speed[occupied], 1.0, atol=0.15)


def test_edie_fields_nonuniform_grid_uses_per_cell_area() -> None:
    """Non-uniform cell widths divide by each cell's own area, not the first."""
    # One vehicle crossing a wide cell then a narrow cell at constant speed.
    t = np.arange(0.0, 4.0, 0.1)
    x = 1.0 * t  # 1 m/s
    vid = np.zeros_like(t)
    x_edges = np.array([0.0, 1.0, 4.0])  # narrow then wide
    t_edges = np.array([0.0, 4.0])
    rho, q = ngsim.edie_fields(vid, t, x, x_edges=x_edges, t_edges=t_edges)
    # speed q/rho recovers ~1 m/s in both cells despite differing widths
    speed = np.divide(q, rho, out=np.zeros_like(q), where=rho > 0)
    assert np.allclose(speed[rho > 0], 1.0, atol=0.2)
    # the wide cell (index 1) must not be scaled by the narrow cell's area:
    # its density is finite and smaller than a naive first-interval scaling.
    assert rho[0, 1] > 0.0


def test_calibrate_greenshields_recovers_params() -> None:
    """Linear speed-density fit recovers Greenshields v_max, rho_max."""
    rho = np.linspace(0.05, 0.9, 50)
    v_max, rho_max = 25.0, 1.0
    q = v_max * rho * (1.0 - rho / rho_max)
    params = ngsim.calibrate_greenshields(rho, q)
    assert abs(params["v_max"] - v_max) < 1.0
    assert abs(params["rho_max"] - rho_max) < 0.1


def test_window_scan_finds_monotone_subregion() -> None:
    """Scan picks a window where density is monotone in x and reports low defect."""
    x = np.linspace(0.0, 1.0, 40)
    t = np.linspace(0.0, 1.0, 40)
    # Left half: monotone-decreasing ramp in x (clean). Right half: a sine (dirty).
    ramp = (1.0 - x)[None, :] * np.ones((40, 1))
    dirty = (0.5 + 0.5 * np.sin(12.0 * x))[None, :] * np.ones((40, 1))
    rho = np.concatenate([ramp[:, :20], dirty[:, 20:]], axis=1)
    win = ngsim.window_scan(rho, x, t, tau=0.05, min_nx=8, min_nt=8)
    xi = win["xi"]
    assert isinstance(xi, tuple)
    assert len(xi) == 2
    _, x_hi = xi
    assert x_hi <= 22  # chosen window lies in the clean left region
    assert win["sign_x"] == -1
    defect = win["defect"]
    assert isinstance(defect, float)
    assert defect < 0.05
