"""Round-trip tests for mononet.core.config."""

from __future__ import annotations

import pytest

from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.types import ActivationSpec, InitSpec


def test_mono_config_roundtrip() -> None:
    cfg = MonoConfig(
        units=8,
        mode="absolute",
        activation=ActivationSpec("elu"),
        convex_fraction=0.25,
        init=InitSpec(scheme="he_normal", seed=3),
        bias=False,
    )
    assert MonoConfig.from_json(cfg.to_json()) == cfg


def test_mono_config_defaults() -> None:
    cfg = MonoConfig(units=4)
    assert cfg.mode == "switch"
    assert cfg.activation.name == "relu"
    assert cfg.convex_fraction == 0.5
    assert cfg.bias is True


def test_mono_config_rejects_bad_units_and_fraction() -> None:
    with pytest.raises(ValueError, match="units must be positive"):
        MonoConfig(units=0)
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoConfig(units=4, convex_fraction=1.5)


def test_mono_residual_config_roundtrip() -> None:
    cfg = MonoResidualConfig(units=16, mode="switch", activation=ActivationSpec("relu"))
    assert MonoResidualConfig.from_json(cfg.to_json()) == cfg
    assert cfg.alpha_gate == "shifted_elu"
    assert cfg.beta_gate == "scaled_elu"
