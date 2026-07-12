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
