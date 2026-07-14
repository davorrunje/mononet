# SPDX-License-Identifier: Apache-2.0
"""Tests for plotting helpers (require matplotlib; skipped without it)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.figure import Figure

from applications.pinn.core import plotting


def test_plotting_returns_figures() -> None:
    """Plot helpers return Figure objects with the expected content."""
    x = np.linspace(0.0, 1.0, 20)
    t = np.linspace(0.0, 1.0, 10)
    fig1 = plotting.profiles(x, {"exact": x, "pred": x**2}, title="p")
    fig2 = plotting.tv_curves(t, {"hard": np.ones_like(t)})
    fig3 = plotting.field_heatmap(np.outer(t, x), x, t)
    assert isinstance(fig1, Figure)
    assert isinstance(fig2, Figure)
    assert isinstance(fig3, Figure)
    assert len(fig1.axes[0].get_lines()) == 2


def test_reconstruction_profile_with_obs() -> None:
    """Reconstruction profile draws each series and overlays observations."""
    x = np.linspace(-1.0, 1.0, 20)
    obs = (np.array([-0.5, 0.5]), np.array([0.2, 0.8]))
    fig = plotting.reconstruction_profile(
        x, {"true": x, "hard": x**3}, obs=obs, title="r"
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].get_lines()) == 2  # two profile curves
    assert len(fig.axes[0].collections) == 1  # one scatter (obs)


def test_metric_vs_noise() -> None:
    """Crossover helper plots one line per method against the noise axis."""
    noise = np.array([0.0, 0.05, 0.1])
    fig = plotting.metric_vs_noise(
        noise, {"hard": noise, "vanilla": 2 * noise}, ylabel="L1 error"
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].get_lines()) == 2


def test_metric_bars() -> None:
    """Grouped bar helper returns a Figure with one bar group per metric."""
    fig = plotting.metric_bars(
        ["hard", "vanilla", "asm"],
        {"L1": [1.0, 2.0, 1.5], "held-out RMSE": [0.1, 0.3, 0.2]},
        ylabel="error",
        title="ngsim",
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes[0].containers) == 2  # two metric groups
