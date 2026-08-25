# SPDX-License-Identifier: Apache-2.0
"""Shared matplotlib theme for application figures.

Kept as a plain rcParams mapping so importing it pulls in no plotting backend;
callers apply it via ``matplotlib.rcParams.update(theme_rc())``.
"""

from __future__ import annotations

from typing import Any


def theme_rc() -> dict[str, Any]:
    """Return the shared matplotlib rcParams overrides for figures.

    :returns: A mapping of rcParam keys to values, suitable for
        `matplotlib.rcParams.update`.
    """
    return {
        "figure.dpi": 120,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 10,
        "figure.autolayout": True,
    }
