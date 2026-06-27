"""Unit tests for mononet.core.types."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask


class TestMonotonicityMask:
    def test_accepts_plus_minus_one(self) -> None:
        mask = MonotonicityMask(np.array([1, -1, 1, -1], dtype=np.int8))
        assert mask.shape == (4,)
        assert len(mask) == 4

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            MonotonicityMask(np.array([1, 0, -1], dtype=np.int8))

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            MonotonicityMask(np.zeros((2, 3), dtype=np.int8))

    def test_is_frozen(self) -> None:
        mask = MonotonicityMask(np.array([1, -1], dtype=np.int8))
        with pytest.raises(dataclasses.FrozenInstanceError):
            mask.values = np.array([1, 1], dtype=np.int8)  # type: ignore[misc]


class TestActivationSpec:
    @pytest.mark.parametrize("name", ["relu", "elu", "selu", "softplus"])
    def test_accepts_a_breve_family(self, name: str) -> None:
        assert ActivationSpec(name=name).name == name  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["tanh", "sigmoid", "frobnicate", "gelu"])
    def test_rejects_bounded_or_unknown(self, name: str) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            ActivationSpec(name=name)  # type: ignore[arg-type]


class TestInitSpec:
    def test_default_is_he_normal(self) -> None:
        assert InitSpec().scheme == "he_normal"
