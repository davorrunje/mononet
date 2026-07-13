# SPDX-License-Identifier: Apache-2.0
"""Legacy MonoDense layer for migration from airtai/monotonic-nn.

A faithful, backend-agnostic reproduction of the original ``airtai/monotonic-nn``
implementation using ``keras.ops`` so it runs under any Keras 3 backend. Reproduces
the original public API (``MonoDense`` plus builders and helpers) as a migration
bridge; new code should use ``mononet.torch`` / ``mononet.jax`` / ``mononet.keras``
instead. Every ``MonoDense`` construction emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import numpy as np
from keras import activations, ops

if TYPE_CHECKING:
    from collections.abc import Callable

_WARNED = False

_DEPRECATION_MESSAGE = (
    "mononet.legacy.MonoDense reproduces the original airtai/monotonic-nn layer "
    "for migration only. Prefer mononet.torch/jax/keras (MonoLinear/MonoDense). "
    "Note the monotonicity spec changed: the legacy {-1, 0, 1} indicator maps to "
    "a two-value +/-1 mask in the new layers."
)


def _warn_once() -> None:
    """Emit the legacy :class:`DeprecationWarning` at most once per process."""
    global _WARNED
    if not _WARNED:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        _WARNED = True


def get_saturated_activation(
    convex_activation: Callable[[Any], Any],
    concave_activation: Callable[[Any], Any],
    a: float = 1.0,
    c: float = 1.0,
) -> Callable[[Any], Any]:
    """Build the saturated activation from a convex/concave pair.

    :param convex_activation: Convex, monotonically increasing base activation.
    :param concave_activation: Its concave reflection ``-f(-x)``.
    :param a: Output scale.
    :param c: Knot location of the piecewise join.
    :returns: A callable mapping a tensor to the saturated activation.
    """

    def saturated_activation(x: Any) -> Any:
        cc = convex_activation(ops.ones_like(x) * c)
        return a * ops.where(
            x <= 0,
            convex_activation(x + c) - cc,
            concave_activation(x - c) + cc,
        )

    return saturated_activation


@lru_cache
def get_activation_functions(
    activation: str | Callable[[Any], Any] | None = None,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any]]:
    """Resolve the convex, concave, and saturated activations for a base name.

    :param activation: Base activation name or callable (assumed convex,
        monotonically increasing). ``None`` resolves to the linear activation.
    :returns: ``(convex_activation, concave_activation, saturated_activation)``.
    """
    convex_activation = activations.get(
        activation.lower() if isinstance(activation, str) else activation
    )

    def concave_activation(x: Any) -> Any:
        return -convex_activation(-x)

    saturated_activation = get_saturated_activation(
        convex_activation, concave_activation
    )
    return convex_activation, concave_activation, saturated_activation


def apply_activations(
    x: Any,
    *,
    units: int,
    convex_activation: Callable[[Any], Any],
    concave_activation: Callable[[Any], Any],
    saturated_activation: Callable[[Any], Any],
    is_convex: bool = False,
    is_concave: bool = False,
    activation_weights: tuple[float, float, float] = (7.0, 7.0, 2.0),
) -> Any:
    """Split ``x`` into convex/concave/saturated groups and activate each.

    :param x: Pre-activation tensor of shape ``(batch, units)``.
    :param units: Output width (the size of the last axis of ``x``).
    :param convex_activation: Convex branch activation.
    :param concave_activation: Concave branch activation.
    :param saturated_activation: Saturated branch activation.
    :param is_convex: Force an all-convex split ``(units, 0, 0)``.
    :param is_concave: Force an all-concave split ``(0, units, 0)``.
    :param activation_weights: Relative sizes of the three groups; ignored when
        ``is_convex`` or ``is_concave`` is set.
    :returns: Activated tensor of shape ``(batch, units)``.
    :raises ValueError: If ``activation_weights`` is not length 3 or has a
        negative entry.
    """
    if convex_activation is None:
        return x

    if is_convex:
        normalized_activation_weights = np.array([1.0, 0.0, 0.0])
    elif is_concave:
        normalized_activation_weights = np.array([0.0, 1.0, 0.0])
    else:
        if len(activation_weights) != 3:
            raise ValueError(f"activation_weights={activation_weights}")
        if (np.array(activation_weights) < 0).any():
            raise ValueError(f"activation_weights={activation_weights}")
        normalized_activation_weights = np.array(activation_weights) / sum(
            activation_weights
        )

    s_convex = round(normalized_activation_weights[0] * units)
    s_concave = round(normalized_activation_weights[1] * units)

    # keras.ops.split takes cut points (numpy semantics), not sizes.
    x_convex, x_concave, x_saturated = ops.split(
        x, [s_convex, s_convex + s_concave], axis=-1
    )

    y_convex = convex_activation(x_convex)
    y_concave = concave_activation(x_concave)
    y_saturated = saturated_activation(x_saturated)

    return ops.concatenate([y_convex, y_concave, y_saturated], axis=-1)


class MonoDense:  # placeholder, replaced in Task 4
    """Placeholder replaced by the real layer in a later task."""
