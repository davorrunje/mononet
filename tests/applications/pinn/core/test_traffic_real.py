# SPDX-License-Identifier: Apache-2.0
"""Tests for the NGSIM-backed ngsim_wave problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from applications.pinn.core.problems import get
from applications.pinn.core.problems.base import Problem

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_fixture(path: Path) -> None:
    x = np.linspace(0.0, 100.0, 12)
    t = np.linspace(0.0, 30.0, 10)
    rho = (0.8 - 0.006 * x)[None, :] * np.ones((10, 1))  # monotone dec in x
    q = 25.0 * rho * (1.0 - rho / 1.0)
    np.savez(
        path,
        x=x,
        t=t,
        rho=rho,
        q=q,
        v_max=25.0,
        rho_max=1.0,
        sign_x=-1,
        monotonicity_defect=0.0,
        provenance="fixture",
    )


def test_ngsim_wave_problem(tmp_path: Path) -> None:
    """The problem loads the npz, satisfies the protocol, interpolates the field."""
    npz = tmp_path / "wave.npz"
    _write_fixture(npz)
    ctor = cast("Callable[..., Problem]", get("ngsim_wave"))
    prob = ctor(npz_path=str(npz))
    assert isinstance(prob, Problem)
    (x0, x1), (_t0, _t1) = prob.domain
    assert (x0, x1) == (0.0, 100.0)
    # ground_truth at a grid node equals the stored value.
    val = prob.ground_truth(np.array([0.0]), np.array([0.0]))
    assert val is not None
    assert abs(float(val[0]) - 0.8) < 1e-6
    # admissibility mask is monotone-decreasing in x.
    assert prob.admissibility().mask == (-1, 0)
    # flux_prime is the smooth Greenshields characteristic speed.
    fp = prob.flux_prime(np.array([0.5]))
    assert abs(float(fp[0]) - 25.0 * (1.0 - 2.0 * 0.5 / 1.0)) < 1e-6
