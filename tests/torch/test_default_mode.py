from __future__ import annotations

import pytest

pytest.importorskip("torch")

from torch import nn

from mononet.torch import MonoLinear, MonoResidual


def test_monolinear_default_mode_is_mixed() -> None:
    """A bare MonoLinear defaults to mixed mode."""
    assert MonoLinear(4, 8).mode == "mixed"


def test_monoresidual_default_mode_is_mixed() -> None:
    """A bare MonoResidual (default F) builds its sublayers in mixed mode."""
    block = MonoResidual(8, 8, activation="elu")
    sub = block.F[0] if isinstance(block.F, nn.Sequential) else block.F
    assert sub.mode == "mixed"
