from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

optuna = pytest.importorskip("optuna")

from benchmarks._common.search_spaces import suggest_config  # noqa: E402

if TYPE_CHECKING:
    from benchmarks._common.config import BenchmarkConfig


def _cfg(
    mode: Literal["switch", "absolute"],
    residual: bool,
    metric: Literal["accuracy", "rmse", "mse"] = "mse",
) -> BenchmarkConfig:
    study = optuna.create_study()
    trial = study.ask()
    return suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode=mode,
        residual=residual,
        epochs=3,
        metric=metric,
    )


def test_absolute_searches_convex_fraction_within_unit_interval() -> None:
    cfg = _cfg("absolute", False, metric="mse")
    assert cfg.mode == "absolute"
    assert cfg.residual is False
    assert 0.0 <= cfg.convex_fraction <= 1.0
    assert cfg.activation == "elu"
    assert cfg.epochs == 3
    assert 1 <= cfg.depth <= 4
    assert cfg.metrics == ("mse",)


def test_switch_uses_fixed_convex_fraction() -> None:
    # switch mode ignores convex_fraction; the sampler must NOT add it as a
    # search dimension (kept at the 0.5 default so studies don't carry a dead param).
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode="switch",
        residual=True,
        epochs=2,
        metric="accuracy",
    )
    assert cfg.mode == "switch"
    assert cfg.residual is True
    assert cfg.convex_fraction == 0.5
    assert "convex_fraction" not in trial.params
    assert cfg.metrics == ("accuracy",)
