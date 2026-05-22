"""Unit tests for mononet.core.types."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mononet.core.types import ActivationSpec, MonotonicityMask


class TestMonotonicityMask:
    def test_accepts_valid_values(self) -> None:
        mask = MonotonicityMask(np.array([1, 0, -1, 0, 1], dtype=np.int8))
        assert mask.shape == (5,)

    def test_rejects_out_of_range_values(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            MonotonicityMask(np.array([2, 0, -1], dtype=np.int8))

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            MonotonicityMask(np.zeros((2, 3), dtype=np.int8))

    def test_is_hashable_and_frozen(self) -> None:
        mask = MonotonicityMask(np.array([1, 0, -1], dtype=np.int8))
        # frozen dataclass: setting attributes must raise FrozenInstanceError
        with pytest.raises(dataclasses.FrozenInstanceError):
            mask.values = np.array([0, 0, 0], dtype=np.int8)  # type: ignore[misc]


class TestActivationSpec:
    @pytest.mark.parametrize("name", ["relu", "tanh", "sigmoid", "elu"])
    def test_accepts_known_activations(self, name: str) -> None:
        spec = ActivationSpec(name=name)  # type: ignore[arg-type]
        assert spec.name == name

    def test_rejects_unknown_activation(self) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            ActivationSpec(name="frobnicate")  # type: ignore[arg-type]
