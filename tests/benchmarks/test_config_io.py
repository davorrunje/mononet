"""Tests for benchmark config IO (TOML loading)."""

from pathlib import Path

from benchmarks._common.config_io import load_config

CONFIGS = Path("benchmarks/configs")


def test_auto_config_values() -> None:
    cfg = load_config(
        CONFIGS / "auto.toml", backend="torch", mode="switch", residual=False
    )
    assert cfg.dataset == "auto"
    assert cfg.depth == 2
    assert cfg.width == 21
    assert cfg.activation == "elu"
    assert cfg.batch_size == 16
    assert cfg.epochs == 50
    assert abs(cfg.optimizer.lr - 0.073407) < 1e-9
    assert cfg.backend == "torch"
    assert cfg.mode == "switch"
    assert cfg.residual is False
