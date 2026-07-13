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


def reconstruction_profile(
    x_values: Array,
    series: Mapping[str, Array],
    *,
    obs: tuple[Array, Array] | None = None,
    title: str = "",
    ylabel: str = "u",
) -> Figure:
    """Spatial profiles at a fixed time, with optional observation scatter.

    The money-shot figure: a monotone reconstruction stays a clean ramp across the
    shock while unconstrained fits oscillate. ``obs`` (if given) overlays the sparse
    noisy points the inverse model was fit to, near this time slice.

    :param x_values: Shared spatial axis.
    :param series: Mapping of label to profile values (the true field first reads
        best). The first entry is drawn as a thick reference curve.
    :param obs: Optional ``(x, u)`` scatter of observations to overlay.
    :param title: Optional axes title.
    :param ylabel: Y-axis label.
    :returns: The figure.
    """
    fig = Figure(figsize=(6.0, 4.0))
    ax = fig.subplots()
    for i, (label, values) in enumerate(series.items()):
        ax.plot(x_values, values, label=label, linewidth=2.4 if i == 0 else 1.4)
    if obs is not None:
        ox, ou = obs
        ax.scatter(
            ox, ou, s=18, c="k", alpha=0.5, zorder=5, label="noisy obs", marker="x"
        )
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
    return fig


def metric_vs_noise(
    noise_values: Array,
    series: Mapping[str, Array],
    *,
    ylabel: str,
    title: str = "",
) -> Figure:
    """Plot a per-method metric against observation-noise level (crossover curve).

    :param noise_values: Noise-std axis.
    :param series: Mapping of method label to metric values at each noise level.
    :param ylabel: Y-axis label (e.g. ``"L1 error"``).
    :param title: Optional axes title.
    :returns: The figure.
    """
    fig = Figure(figsize=(5.5, 4.0))
    ax = fig.subplots()
    for label, values in series.items():
        ax.plot(noise_values, values, marker="o", label=label)
    ax.set_xlabel("observation noise (std)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)
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
