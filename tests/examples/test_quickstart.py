# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: each landing-page quickstart module builds and runs."""

from __future__ import annotations

import os

import pytest

from tests.examples._loader import load_example


def test_quickstart_torch() -> None:
    pytest.importorskip("torch")
    mod = load_example("quickstart_torch.py")
    assert tuple(mod.y.shape) == (8, 1)


def test_quickstart_jax() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax.nnx")
    mod = load_example("quickstart_jax.py")
    assert tuple(mod.y.shape) == (8, 1)


def test_quickstart_keras() -> None:
    os.environ.setdefault("KERAS_BACKEND", "jax")
    pytest.importorskip("keras")
    mod = load_example("quickstart_keras.py")
    assert tuple(mod.y.shape) == (8, 1)
