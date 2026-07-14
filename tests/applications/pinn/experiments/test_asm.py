# SPDX-License-Identifier: Apache-2.0
"""Adaptive Smoothing Method baseline reconstructs a moving wave from sparse data."""

from __future__ import annotations

import numpy as np

from applications.pinn.experiments.baselines import adaptive_smoothing


def test_asm_reconstructs_moving_front() -> None:
    """ASM recovers a front advecting at the congested wave speed from few detectors."""
    x = np.linspace(0.0, 1000.0, 50)
    t = np.linspace(0.0, 100.0, 40)
    c = -5.0  # congested wave speed (m/s), backward-moving
    gx, gt = np.meshgrid(x, t)
    truth = 1.0 / (1.0 + np.exp((gx - (500.0 + c * gt)) / 20.0))  # monotone front
    # sparse detectors: 5 fixed x-lines
    xi = np.linspace(2, 47, 5).astype(int)
    oc = np.vstack([np.column_stack([np.full(len(t), x[i]), t]) for i in xi])
    ov = np.concatenate([truth[:, i] for i in xi])
    rec = adaptive_smoothing(oc, ov, x, t, v_free=25.0, v_cong=c, sigma=40.0, tau=8.0)
    rmse = float(np.sqrt(np.mean((rec - truth) ** 2)))
    assert rmse < 0.15
