# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic loss configuration and training-data container.

The PDE residual and its derivatives are backend-specific (they use autodiff),
so they live in the per-backend trainers. What is shared — and identical across
methods and backends — is the *weighting* of the loss terms and the *data* the
loss is evaluated on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.floating]
Supervised = tuple[Array, Array]  # (coords (N, 2) [x, t], values (N,))


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Relative weights of the PINN loss terms.

    :param residual: PDE-residual term.
    :param ic: Initial-condition term (forward tier).
    :param bc: Boundary-condition term (forward tier).
    :param data: Observation data-fit term (inverse tier).
    :param mono: Soft monotonicity-penalty weight — **nonzero only for the
        ``soft`` baseline**; the architecture supplies monotonicity otherwise.
    """

    residual: float = 1.0
    ic: float = 1.0
    bc: float = 1.0
    data: float = 1.0
    mono: float = 0.0


@dataclass(frozen=True, slots=True)
class TrainingData:
    """Point sets a PINN trains on (NumPy; each backend converts to its tensors).

    :param collocation: Interior points ``(N, 2)`` where the residual is enforced.
    :param ic: Optional ``(coords, values)`` on the initial line (forward tier).
    :param bc: Optional ``(coords, values)`` on the boundaries (forward tier).
    :param obs: Optional ``(coords, values)`` sparse observations (inverse tier).
    """

    collocation: Array
    ic: Supervised | None = None
    bc: Supervised | None = None
    obs: Supervised | None = None
