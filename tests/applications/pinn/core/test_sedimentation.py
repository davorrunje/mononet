# SPDX-License-Identifier: Apache-2.0
"""Tests for the batch-sedimentation (Kynch) problem."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from applications.pinn.core.problems import get
from applications.pinn.core.problems.base import Problem

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_fixture(path: Path) -> None:
    """Write a tiny synthetic settling field: C monotone-decreasing with height z."""
    z = np.linspace(0.0, 1.0, 12)  # column height (m)
    t = np.linspace(0.0, 60.0, 10)  # time (min)
    # dense at the bottom (z=0), clear at the top (z=1): monotone decreasing in z
    c = (10.0 * (1.0 - z))[None, :] * np.ones((10, 1))
    np.savez(path, x=z, t=t, rho=c, v0=1.0e-3, c_max=12.0, n=2.0, sign_x=-1)


def test_sediment_batch_problem(tmp_path: Path) -> None:
    """Loads the field, satisfies the protocol, interpolates, uses the Kynch flux."""
    npz = tmp_path / "batch.npz"
    _write_fixture(npz)
    ctor = cast("Callable[..., Problem]", get("sediment_batch"))
    prob = ctor(npz_path=str(npz))
    assert isinstance(prob, Problem)

    (z0, z1), (t0, t1) = prob.domain
    assert (z0, z1) == (0.0, 1.0)
    assert (t0, t1) == (0.0, 60.0)

    # ground_truth at a grid node equals the stored value (C=10 at z=0).
    val = prob.ground_truth(np.array([0.0]), np.array([0.0]))
    assert val is not None
    assert abs(float(val[0]) - 10.0) < 1e-6

    # admissibility: monotone-decreasing in height.
    assert prob.admissibility().mask == (-1, 0)

    # Kynch flux vanishes at c=0 and c=c_max, positive between.
    assert abs(float(prob.flux(np.array([0.0]))[0])) < 1e-12
    assert abs(float(prob.flux(np.array([12.0]))[0])) < 1e-12
    assert float(prob.flux(np.array([6.0]))[0]) > 0.0
