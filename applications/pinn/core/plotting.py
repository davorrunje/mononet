# SPDX-License-Identifier: Apache-2.0
"""Matplotlib figures for the PINN paper.

Figures are built with the object-oriented API (no ``pyplot`` global state, no
GUI backend), so the functions are pure and safe to call in tests and notebooks.
Callers save or embed the returned :class:`~matplotlib.figure.Figure`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from collections.abc import Mapping

Array = npt.NDArray[np.floating]


def profiles(
    x_values: Array,
    series: Mapping[str, Array],
    *,
    title: str = "",
) -> Figure:
    """Plot several named solution profiles against ``x``.

    :param x_values: Shared spatial axis.
    :param series: Mapping of label to profile values.
    :param title: Optional axes title.
    :returns: The figure.
    """
    fig = Figure()
    ax = fig.subplots()
    for label, values in series.items():
        ax.plot(x_values, values, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    if title:
        ax.set_title(title)
    ax.legend()
    return fig


def tv_curves(
    t_values: Array,
    curves: Mapping[str, Array],
    *,
    title: str = "",
) -> Figure:
    """Plot total-variation-vs-time curves for several methods.

    :param t_values: Time axis.
    :param curves: Mapping of label to TV(t) values.
    :param title: Optional axes title.
    :returns: The figure.
    """
    fig = Figure()
    ax = fig.subplots()
    for label, values in curves.items():
        ax.plot(t_values, values, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("TV(t)")
    if title:
        ax.set_title(title)
    ax.legend()
    return fig


def field_heatmap(
    field: Array,
    x_values: Array,
    t_values: Array,
    *,
    title: str = "",
) -> Figure:
    """Render a space-time field as a heatmap.

    :param field: Field of shape ``(len(t_values), len(x_values))``.
    :param x_values: Spatial axis.
    :param t_values: Temporal axis.
    :param title: Optional axes title.
    :returns: The figure.
    """
    fig = Figure()
    ax = fig.subplots()
    extent = (
        float(x_values[0]),
        float(x_values[-1]),
        float(t_values[0]),
        float(t_values[-1]),
    )
    image = ax.imshow(
        field, origin="lower", aspect="auto", extent=extent, cmap="viridis"
    )
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    if title:
        ax.set_title(title)
    return fig
