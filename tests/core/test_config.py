"""Unit tests for mononet.core.config."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mononet.core.config import MonoLinearConfig
from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask


def _mask(n: int) -> MonotonicityMask:
    return MonotonicityMask(np.zeros(n, dtype=np.int8))


class TestMonoLinearConfig:
    def test_constructs_with_valid_args(self) -> None:
        cfg = MonoLinearConfig(
            in_features=8,
            out_features=4,
            monotonicity=_mask(8),
            activation=ActivationSpec(name="relu"),
            init=InitSpec(),
        )
        assert cfg.in_features == 8
        assert cfg.out_features == 4

    def test_rejects_non_positive_in_features(self) -> None:
        with pytest.raises(ValueError, match="in_features must be positive"):
            MonoLinearConfig(
                in_features=0,
                out_features=4,
                monotonicity=_mask(8),
                activation=ActivationSpec(name="relu"),
                init=InitSpec(),
            )

    def test_rejects_non_positive_out_features(self) -> None:
        with pytest.raises(ValueError, match="out_features must be positive"):
            MonoLinearConfig(
                in_features=8,
                out_features=-1,
                monotonicity=_mask(8),
                activation=ActivationSpec(name="relu"),
                init=InitSpec(),
            )

    def test_rejects_mismatched_mask_length(self) -> None:
        with pytest.raises(ValueError, match="mask length"):
            MonoLinearConfig(
                in_features=8,
                out_features=4,
                monotonicity=_mask(7),
                activation=ActivationSpec(name="relu"),
                init=InitSpec(),
            )

    def test_round_trips_through_json(self) -> None:
        cfg = MonoLinearConfig(
            in_features=8,
            out_features=4,
            monotonicity=MonotonicityMask(
                np.array([1, 1, 0, 0, -1, -1, 0, 0], dtype=np.int8)
            ),
            activation=ActivationSpec(name="tanh"),
            init=InitSpec(scheme="he_normal", seed=42),
        )
        payload = cfg.to_json()
        d = json.loads(payload)
        assert d["in_features"] == 8
        round_tripped = MonoLinearConfig.from_json(payload)
        assert round_tripped.in_features == cfg.in_features
        assert round_tripped.out_features == cfg.out_features
        assert round_tripped.activation.name == "tanh"
        assert round_tripped.init.seed == 42
        np.testing.assert_array_equal(
            round_tripped.monotonicity.values, cfg.monotonicity.values
        )
