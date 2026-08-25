# SPDX-License-Identifier: Apache-2.0
"""Admissibility specification and violation metric.

The *admissibility condition* of a structure-preserving PINN is a monotonicity
(and, for later papers, convexity) statement about the solution field. This
module holds the declarative spec and the non-negative violation functional that
is the paper's headline metric: it is ``0`` by construction for the hard-monotone
model and ``> 0`` for unconstrained / soft-penalty baselines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class AdmissibilitySpec:
    """Declarative admissibility condition for a problem's solution field.

    :param mask: Monotonicity sign per input axis — ``+1`` non-decreasing,
        ``-1`` non-increasing, ``0`` unconstrained (mirrors
        `mononet.MonotonicityMask`).
    :param convex_axes: Input axes in which the solution is convex.
    :param concave_axes: Input axes in which the solution is concave.
    """

    mask: tuple[int, ...]
    convex_axes: tuple[int, ...] = ()
    concave_axes: tuple[int, ...] = ()


def violation(
    field: npt.NDArray[np.floating],
    *,
    axis: int,
    sign: int,
) -> float:
    """Total wrong-sign variation of ``field`` along ``axis``.

    For a non-increasing target (``sign == -1``) every *upward* step is a
    violation; the returned value is the summed magnitude of those steps — i.e.
    the total mass by which the field fails to be monotone. It is ``0`` iff the
    field is monotone in the required direction along ``axis``.

    :param field: Solution field sampled on a grid.
    :param axis: Axis along which monotonicity is required.
    :param sign: ``-1`` non-increasing, ``+1`` non-decreasing, ``0``
        unconstrained (always ``0``).
    :returns: Non-negative total wrong-sign variation.
    """
    if sign == 0:
        return 0.0
    diffs = np.diff(field, axis=axis)
    wrong = np.maximum(diffs, 0.0) if sign == -1 else np.maximum(-diffs, 0.0)
    return float(wrong.sum())
