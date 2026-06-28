"""CI smoke test: build + train 2 epochs on a synthetic bundle for one flavor."""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import pytest

BACKEND: Literal["torch", "jax", "keras"] = os.environ.get(  # type: ignore[assignment]
    "MONONET_TEST_BACKEND", "torch"
)
pytest.importorskip(BACKEND if BACKEND != "keras" else "keras")

from benchmarks._common.bundle import DatasetBundle  # noqa: E402
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec  # noqa: E402
from benchmarks._common.runner import run  # noqa: E402


def test_smoke_one_flavor_trains() -> None:
    """Train 2 epochs on a synthetic in-memory bundle and assert a finite ResultRow."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 5)).astype(np.float64)
    y = (X[:, 0] > 0).astype(np.float64)
    b = DatasetBundle(
        "syn",
        "binary_classification",
        X,
        y,
        X,
        y,
        (0,),
        (),
        tuple(f"f{i}" for i in range(5)),
        {},
    )
    cfg = BenchmarkConfig(
        dataset="syn",
        backend=BACKEND,
        mode="switch",
        residual=False,
        depth=1,
        width=8,
        activation="elu",
        convex_fraction=0.5,
        embed_hidden=(8,),
        dropout=0.0,
        optimizer=OptimizerSpec("adam", 1e-2, 0.0),
        lr_decay=None,
        batch_size=32,
        epochs=2,
        early_stopping=None,
        seeds=(0,),
        metrics=("accuracy",),
    )
    rows = run(cfg, b)
    assert len(rows) == 1
    assert np.isfinite(rows[0].scores["accuracy"])
