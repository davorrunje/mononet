"""Tests that pin the public signature of the NumPy reference."""

from __future__ import annotations

import inspect

from mononet.core import reference


def test_monotonic_dense_signature() -> None:
    sig = inspect.signature(reference.monotonic_dense)
    assert list(sig.parameters) == [
        "x",
        "weights",
        "bias",
        "mode",
        "activation",
        "convex_fraction",
    ]


def test_monotonic_mlp_removed() -> None:
    assert not hasattr(reference, "monotonic_mlp")
