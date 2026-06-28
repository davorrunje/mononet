"""Integration test for benchmarks._common.runner.run()."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.runner import run


def _bundle(n: int = 128, d: int = 6) -> DatasetBundle:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = (X[:, 0] + 0.1 * rng.normal(size=n)).astype(np.float64)
    return DatasetBundle(
        "syn",
        "regression",
        X,
        y,
        X,
        y,
        (0,),
        (),
        tuple(f"f{i}" for i in range(d)),
        {},
    )


def test_run_returns_one_row_per_seed_with_finite_metric() -> None:
    cfg = BenchmarkConfig(
        dataset="syn",
        backend="torch",
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
        epochs=3,
        early_stopping=None,
        seeds=(0, 1),
        metrics=("mse",),
    )
    rows = run(cfg, _bundle())
    assert len(rows) == 2
    assert all(np.isfinite(r.scores["mse"]) for r in rows)
    assert all(r.seed in (0, 1) for r in rows)
