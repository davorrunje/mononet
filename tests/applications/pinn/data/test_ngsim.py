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


def test_calibrate_greenshields_recovers_params() -> None:
    """Linear speed-density fit recovers Greenshields v_max, rho_max."""
    rho = np.linspace(0.05, 0.9, 50)
    v_max, rho_max = 25.0, 1.0
    q = v_max * rho * (1.0 - rho / rho_max)
    params = ngsim.calibrate_greenshields(rho, q)
    assert abs(params["v_max"] - v_max) < 1.0
    assert abs(params["rho_max"] - rho_max) < 0.1
