# SPDX-License-Identifier: Apache-2.0
"""Round-trip tests for mononet.core.config."""

from __future__ import annotations

import pytest

from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.types import ActivationSpec, InitSpec


def test_mono_config_roundtrip() -> None:
    cfg = MonoConfig(
        units=8,
        mode="mixed",
        activation=ActivationSpec("elu"),
        convex_fraction=0.25,
        init=InitSpec(scheme="he_normal", seed=3),
        bias=False,
    )
    assert MonoConfig.from_json(cfg.to_json()) == cfg


def test_mono_config_defaults() -> None:
    cfg = MonoConfig(units=4)
    assert cfg.mode == "mixed"
    assert cfg.activation.name == "identity"
    assert cfg.convex_fraction == 0.5
    assert cfg.bias is True


def test_mono_config_rejects_bad_units_and_fraction() -> None:
    with pytest.raises(ValueError, match="units must be positive"):
        MonoConfig(units=0)
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoConfig(units=4, convex_fraction=1.5)


def test_mono_residual_config_roundtrip() -> None:
    cfg = MonoResidualConfig(units=16, mode="split", activation=ActivationSpec("relu"))
    assert MonoResidualConfig.from_json(cfg.to_json()) == cfg
    assert cfg.alpha_gate == "shifted_elu"
    assert cfg.beta_gate == "softplus"
    assert cfg.near_zero_scale == pytest.approx(1e-3)
    assert MonoResidualConfig.from_json(
        MonoResidualConfig(
            units=16,
            mode="split",
            activation=ActivationSpec("relu"),
            near_zero_scale=5e-3,
        ).to_json()
    ).near_zero_scale == pytest.approx(5e-3)


def test_monoconfig_default_activation_is_identity() -> None:
    from mononet.core.config import MonoConfig

    assert MonoConfig(units=8).activation.name == "identity"


def test_monoresidualconfig_requires_activation() -> None:
    import pytest

    from mononet.core.config import MonoResidualConfig

    with pytest.raises(TypeError):
        MonoResidualConfig(units=8)  # type: ignore[call-arg]


def test_mono_config_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        MonoConfig(units=4, mode="bogus")  # type: ignore[arg-type]


def test_mono_residual_config_rejects_bad_units_and_mode() -> None:
    with pytest.raises(ValueError, match="units must be positive"):
        MonoResidualConfig(units=0, activation=ActivationSpec("relu"))
    with pytest.raises(ValueError, match="mode must be"):
        MonoResidualConfig(units=4, mode="bogus", activation=ActivationSpec("relu"))  # type: ignore[arg-type]


@pytest.mark.parametrize(("old", "new"), [("absolute", "mixed"), ("switch", "split")])
def test_old_mode_names_rejected_with_hint(old: str, new: str) -> None:
    with pytest.raises(ValueError, match=new):
        MonoConfig(units=4, mode=old)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=new):
        MonoResidualConfig(
            units=4,
            mode=old,  # type: ignore[arg-type]
            activation=ActivationSpec("relu"),
        )


def test_from_dict_rejects_old_mode() -> None:
    with pytest.raises(ValueError, match="mixed"):
        MonoConfig.from_dict(
            {
                "units": 4,
                "mode": "absolute",
                "activation": {"name": "relu"},
                "convex_fraction": 0.5,
                "init": {"scheme": "he_normal", "seed": None},
                "bias": True,
            }
        )
