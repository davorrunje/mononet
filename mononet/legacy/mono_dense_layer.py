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


class MonoDense:  # placeholder, replaced in Task 4
    """Placeholder replaced by the real layer in a later task."""
