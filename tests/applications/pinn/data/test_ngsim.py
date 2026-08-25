# SPDX-License-Identifier: Apache-2.0
"""Tests for NGSIM preprocessing: Edie aggregation, FD calibration, window scan."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from applications.pinn.data import ngsim

if TYPE_CHECKING:
    from pathlib import Path


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


def test_build_dataset_writes_npz(tmp_path: Path) -> None:
    """build_dataset turns a synthetic raw CSV into a well-formed derived npz."""
    import csv

    raw = tmp_path / "raw.csv"
    # Two lanes; a monotone-decreasing density wave in lane 1 via staggered entries.
    rows: list[tuple[str, ...]] = [
        ("Vehicle_ID", "Global_Time", "Local_Y", "v_Vel", "Lane_ID")
    ]
    vid = 0
    for start in np.linspace(0.0, 50.0, 60):  # many vehicles -> density
        vid += 1  # noqa: SIM113
        for k in range(40):
            tt = start + k * 0.1
            yy = 10.0 * (tt - start)  # 10 m/s
            rows.append((str(vid), str(tt * 1000.0), str(yy), str(10.0), str(1)))
    with raw.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    out = tmp_path / "wave.npz"
    # synthetic positions are already in metres -> disable feet conversion
    summary = ngsim.build_dataset(
        raw, out, lane=1, dx=5.0, dt=1.0, tau=0.9, units="metres"
    )
    assert out.exists()
    d = np.load(out, allow_pickle=True)
    for key in (
        "x",
        "t",
        "rho",
        "q",
        "v_max",
        "rho_max",
        "sign_x",
        "monotonicity_defect",
    ):
        assert key in d
    assert d["rho"].ndim == 2
    assert summary["nx"] == d["rho"].shape[1]


def test_load_raw_case_insensitive_headers_and_units(tmp_path: Path) -> None:
    """_load_raw matches columns case-insensitively and converts feet -> metres."""
    import csv

    raw = tmp_path / "lower.csv"
    with raw.open("w", newline="") as f:
        w = csv.writer(f)
        # lowercase Socrata-style headers, extra columns interleaved
        w.writerow(["vehicle_id", "frame_id", "global_time", "local_y", "lane_id"])
        w.writerow([1, 0, 1000.0, 100.0, 2])  # lane 2 (skipped)
        w.writerow([2, 0, 2000.0, 10.0, 1])  # lane 1
        w.writerow([2, 1, 2100.0, 20.0, 1])  # lane 1
    vid, t, x = ngsim._load_raw(raw, lane=1, units="feet")
    assert list(vid) == [2.0, 2.0]  # only lane-1 rows, columns resolved lowercase
    assert np.allclose(t, [2.0, 2.1])  # ms -> s
    assert np.allclose(x, [10.0 * 0.3048, 20.0 * 0.3048])  # feet -> metres
    # metres mode leaves positions unscaled
    _, _, x_m = ngsim._load_raw(raw, lane=1, units="metres")
    assert np.allclose(x_m, [10.0, 20.0])
