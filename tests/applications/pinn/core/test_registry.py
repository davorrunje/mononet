# SPDX-License-Identifier: Apache-2.0
"""Tests for the problem registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from applications.pinn.core.admissibility import AdmissibilitySpec
from applications.pinn.core.problems import base

if TYPE_CHECKING:
    from collections.abc import Iterator

    from applications.pinn.core.problems.base import Array


class _StubProblem:
    """Minimal `Problem`-conforming stub for registry tests."""

    key = "stub"

    @property
    def domain(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return ((0.0, 1.0), (0.0, 1.0))

    def admissibility(self) -> AdmissibilitySpec:
        return AdmissibilitySpec(mask=(-1, 0))

    def flux(self, u: Array) -> Array:
        return u

    def flux_prime(self, u: Array) -> Array:
        return np.ones_like(u)

    def initial(self, x: Array) -> Array:
        return x

    def ground_truth(self, x: Array, t: Array) -> Array | None:
        return None


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Snapshot and restore the module registry around each test."""
    saved = dict(base._REGISTRY)
    yield
    base._REGISTRY.clear()
    base._REGISTRY.update(saved)


def test_register_get_available_roundtrip() -> None:
    """A registered class is retrievable and listed."""

    @base.register("dummy")
    class _Dummy(_StubProblem):
        key = "dummy"

    assert base.get("dummy") is _Dummy
    assert "dummy" in base.available()


def test_get_unknown_raises() -> None:
    """Requesting an unregistered key raises KeyError."""
    with pytest.raises(KeyError):
        base.get("does-not-exist")


def test_duplicate_registration_raises() -> None:
    """Re-registering the same key raises KeyError."""

    @base.register("dup")
    class _A(_StubProblem):
        key = "dup"

    with pytest.raises(KeyError):

        @base.register("dup")
        class _B(_StubProblem):
            key = "dup"


def test_available_is_sorted() -> None:
    """`available` returns keys in sorted order."""

    @base.register("zeta")
    class _Z(_StubProblem):
        key = "zeta"

    @base.register("alpha")
    class _A(_StubProblem):
        key = "alpha"

    keys = base.available()
    assert keys == sorted(keys)
