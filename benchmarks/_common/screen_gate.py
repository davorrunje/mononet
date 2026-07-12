"""Screen verdict gate: does a dataset advance to a full ladder study."""

from __future__ import annotations

from typing import Literal

DEFAULT_MARGIN = 0.005
"""Practical-significance floor on Δ for classification (accuracy)."""


def gate(
    delta_lo: float, delta_point: float, margin: float = DEFAULT_MARGIN
) -> Literal["ladder", "standard"]:
    """Route a dataset by its max-size deep-shallow gap.

    :param delta_lo: Lower bound of the 95% seed-bootstrap band on Δ.
    :param delta_point: Point estimate of Δ = IQM(deep) - IQM(shallow).
    :param margin: Practical-significance floor.
    :returns: ``ladder`` iff Δ is significantly (``delta_lo > 0``) *and*
        practically (``delta_point >= margin``) positive; else ``standard``.
    """
    if delta_lo > 0.0 and delta_point >= margin:
        return "ladder"
    return "standard"
