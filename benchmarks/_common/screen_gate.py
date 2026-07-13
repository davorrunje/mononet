"""Screen verdict gate: does a dataset advance to a full ladder study."""

from __future__ import annotations

from typing import Literal

DEFAULT_MARGIN = 0.005
"""Practical-significance floor on Δ for classification (accuracy)."""


def gate(
    delta_lo: float, delta_point: float, margin: float = DEFAULT_MARGIN
) -> Literal["ladder", "standard"]:
    """Route a dataset by its max-size deep-shallow gap.

    Metric-agnostic: ``delta_lo``/``delta_point`` must already be sign-normalized
    so that positive means "deep is better" (e.g. via
    :func:`benchmarks._common.stage2_gate._signed_improvement`), regardless of
    whether the underlying metric is lower-is-better (e.g. MSE) or
    higher-is-better (e.g. accuracy/AUC). ``margin`` is on that normalized
    scale. Reused as-is by :func:`benchmarks._common.stage2_gate.verdict`.

    :param delta_lo: Lower bound of the 95% seed-bootstrap band on the signed Δ.
    :param delta_point: Point estimate of the signed Δ = IQM(deep) - IQM(shallow).
    :param margin: Practical-significance floor.
    :returns: ``ladder`` iff Δ is significantly (``delta_lo > 0``) *and*
        practically (``delta_point >= margin``) positive; else ``standard``.
    """
    if delta_lo > 0.0 and delta_point >= margin:
        return "ladder"
    return "standard"
