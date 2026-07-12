# SPDX-License-Identifier: Apache-2.0
"""The ``Problem`` protocol and the problem registry.

Each PDE family is a plug-in module that defines one or more `Problem`
implementations and registers them by name. Downstream code (models, trainers,
experiments) discovers problems through :func:`get` / :func:`available` and never
imports the concrete classes directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

    from applications.pinn.core.admissibility import AdmissibilitySpec

Array = npt.NDArray[np.floating]


@runtime_checkable
class Problem(Protocol):
    """A scalar PDE problem on a 1-D spatial domain over time.

    Concrete problems are constructed with their parameters (flux constants,
    Riemann states, …) and expose a uniform interface so the framework can build
    residuals, enforce admissibility, and score against ground truth.
    """

    #: Registry key (class-level identifier, e.g. ``"burgers"``).
    key: str

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return ``((x_min, x_max), (t_min, t_max))``."""

    def admissibility(self) -> AdmissibilitySpec:
        """Return the monotonicity/convexity spec for the solution field."""

    def flux(self, u: Array) -> Array:
        """Evaluate the flux ``f(u)``."""

    def flux_prime(self, u: Array) -> Array:
        """Evaluate the flux derivative ``f'(u)`` (characteristic speed)."""

    def initial(self, x: Array) -> Array:
        """Evaluate the initial condition ``u(x, 0)`` (forward mode)."""

    def ground_truth(self, x: Array, t: Array) -> Array | None:
        """Return the reference solution on ``(x, t)`` if available, else None."""


_REGISTRY: dict[str, type[Problem]] = {}


def register(key: str) -> Callable[[type[Problem]], type[Problem]]:
    """Class decorator registering a `Problem` implementation under ``key``.

    :param key: Unique registry name.
    :returns: The decorator.
    :raises KeyError: If ``key`` is already registered.
    """

    def decorate(cls: type[Problem]) -> type[Problem]:
        if key in _REGISTRY:
            raise KeyError(f"problem {key!r} already registered")
        _REGISTRY[key] = cls
        return cls

    return decorate


def get(key: str) -> type[Problem]:
    """Return the registered `Problem` class for ``key``.

    :param key: Registry name.
    :returns: The problem class.
    :raises KeyError: If ``key`` is not registered.
    """
    if key not in _REGISTRY:
        raise KeyError(f"unknown problem {key!r}; available: {available()}")
    return _REGISTRY[key]


def available() -> list[str]:
    """Return the sorted list of registered problem keys."""
    return sorted(_REGISTRY)
