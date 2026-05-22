"""Tests that pin the public signature of the NumPy reference."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from mononet.core import reference
from mononet.core.types import ActivationSpec, MonotonicityMask


def test_monotonic_dense_signature() -> None:
    sig = inspect.signature(reference.monotonic_dense)
    assert list(sig.parameters) == [
        "x",
        "weights",
        "bias",
        "mask",
        "activation",
    ]


def test_monotonic_mlp_signature() -> None:
    sig = inspect.signature(reference.monotonic_mlp)
    assert list(sig.parameters) == [
        "x",
        "weights",
        "biases",
        "mask",
        "activation",
    ]


def test_monotonic_dense_raises_not_implemented() -> None:
    x = np.zeros((1, 2), dtype=np.float32)
    w = np.zeros((2, 1), dtype=np.float32)
    b = np.zeros((1,), dtype=np.float32)
    with pytest.raises(NotImplementedError):
        reference.monotonic_dense(
            x,
            w,
            b,
            MonotonicityMask(np.zeros(2, dtype=np.int8)),
            ActivationSpec(name="relu"),
        )
