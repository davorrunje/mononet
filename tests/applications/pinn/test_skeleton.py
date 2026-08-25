# SPDX-License-Identifier: Apache-2.0
"""Skeleton smoke tests for the applications.pinn package."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np

from applications._common import metrics_io, seeding

if TYPE_CHECKING:
    from pathlib import Path


def test_import_pinn_does_not_import_backends() -> None:
    """Importing `applications.pinn` must not eagerly import torch or jax.

    Run in a fresh interpreter so the check is independent of test order
    (other tests in the suite may already have imported a backend).
    """
    code = (
        "import sys, applications.pinn\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'jax' not in sys.modules, 'jax was imported'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_rng_is_reproducible() -> None:
    """`seeding.rng` is deterministic per seed and varies across seeds."""
    a = seeding.rng(0).standard_normal(5)
    b = seeding.rng(0).standard_normal(5)
    c = seeding.rng(1).standard_normal(5)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_split_seeds_are_deterministic_and_distinct() -> None:
    """`split_seeds` yields the requested count, deterministically."""
    s1 = seeding.split_seeds(0, 4)
    s2 = seeding.split_seeds(0, 4)
    assert s1 == s2
    assert len(s1) == 4
    assert len(set(s1)) == 4


def test_metrics_io_roundtrips(tmp_path: Path) -> None:
    """`write_result`/`read_result` round-trip a JSON mapping exactly."""
    obj = {"metric": "l2", "value": 1.5, "seeds": [0, 1, 2]}
    path = tmp_path / "nested" / "result.json"
    metrics_io.write_result(path, obj)
    assert metrics_io.read_result(path) == obj
