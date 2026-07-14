"""Divergence flag on ResultRow + early-stopping wiring in the torch runner."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, EarlyStoppingSpec, OptimizerSpec
from benchmarks._common.results import ResultRow
from benchmarks._common.runner import is_diverged, run


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (256, 3)).astype("float32")
    y = x.sum(1).astype("float32")
    return DatasetBundle(
        name="t",
        task="regression",
        X_train=x,
        y_train=y,
        X_test=x,
        y_test=y,
        mono_increasing=(0, 1, 2),
        mono_decreasing=(),
        feature_names=("a", "b", "c"),
        metadata={},
    )


def _cfg(**over: object) -> BenchmarkConfig:
    base: dict[str, object] = {
        "dataset": "t",
        "backend": "torch",
        "mode": "mixed",
        "residual": False,
        "depth": 2,
        "width": 16,
        "activation": "elu",
        "convex_fraction": 0.5,
        "embed_hidden": (),
        "dropout": 0.0,
        "optimizer": OptimizerSpec(name="adam", lr=1e-2),
        "lr_decay": None,
        "batch_size": 64,
        "epochs": 8,
        "early_stopping": None,
        "seeds": (0,),
        "metrics": ("mse",),
    }
    base.update(over)
    return BenchmarkConfig(**base)  # type: ignore[arg-type]


def test_is_diverged_predicate() -> None:
    assert is_diverged(float("nan"), 1.0) is True
    assert is_diverged(float("inf"), 1.0) is True
    assert is_diverged(0.5, 1.0) is False
    assert is_diverged(20.0, 1.0) is True
    assert is_diverged(9.9, 1.0) is False


def test_result_row_has_diverged_field_and_roundtrips() -> None:
    row = ResultRow(
        dataset="t",
        backend="torch",
        mode="mixed",
        residual=False,
        seed=0,
        scores={"mse": 0.1},
        epochs_run=5,
        diverged=True,
    )
    d = dataclasses.asdict(row)
    assert d["diverged"] is True
    assert ResultRow(**d).diverged is True


def test_normal_run_is_not_diverged() -> None:
    rows = run(_cfg(), _bundle())
    assert len(rows) == 1
    assert rows[0].diverged is False


def test_early_stopping_stops_before_max_epochs() -> None:
    # trivial linear-monotone target converges fast; with a relative min_delta
    # the patience counter fires once gains stall, well before the 300 ceiling.
    cfg = _cfg(
        epochs=300,
        early_stopping=EarlyStoppingSpec(monitor="val", patience=5, min_delta=1e-3),
    )
    rows = run(cfg, _bundle())
    assert rows[0].epochs_run < 300


def test_min_delta_fires_where_zero_delta_would_not() -> None:
    # a positive relative min_delta must stop no later than a zero one — with a
    # slowly-improving loss, min_delta=0 can run to the ceiling while min_delta>0
    # stops once relative gains fall below the threshold.
    base: dict[str, object] = {"epochs": 300, "seeds": (0,)}
    lax = _cfg(**base, early_stopping=EarlyStoppingSpec("val", 5, min_delta=0.0))
    strict = _cfg(**base, early_stopping=EarlyStoppingSpec("val", 5, min_delta=1e-2))
    assert run(strict, _bundle())[0].epochs_run <= run(lax, _bundle())[0].epochs_run


def test_no_early_stopping_runs_full_epochs() -> None:
    cfg = _cfg(epochs=8, early_stopping=None)
    rows = run(cfg, _bundle())
    assert rows[0].epochs_run == 8


def test_early_stopping_run_not_falsely_diverged() -> None:
    # regression: a cleanly-converging run with early stopping must NOT be
    # flagged diverged just because early-epoch val loss exceeds the baseline.
    cfg = _cfg(
        epochs=200,
        early_stopping=EarlyStoppingSpec(monitor="val", patience=10, min_delta=1e-3),
    )
    rows = run(cfg, _bundle())
    assert rows[0].diverged is False
