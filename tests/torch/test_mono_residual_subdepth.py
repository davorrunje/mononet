from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear, MonoResidual  # noqa: E402


def test_default_builds_two_monolinears() -> None:
    layer = MonoResidual(
        8, 8, mode="mixed", activation="elu"
    )  # default sub_depth -> 2
    assert isinstance(layer.F, torch.nn.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F) == 2


def test_subdepth_builds_k_monolinears() -> None:
    layer = MonoResidual(8, 8, mode="mixed", activation="elu", sub_depth=3)
    assert isinstance(layer.F, torch.nn.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F) == 3


def test_subdepth1_is_single_monolinear() -> None:
    layer = MonoResidual(8, 8, mode="mixed", activation="elu", sub_depth=1)
    assert isinstance(layer.F, MonoLinear)


def test_F_alone_is_used() -> None:  # noqa: N802  # F without sub_depth must NOT raise
    f = MonoLinear(8, 8, mode="mixed")
    layer = MonoResidual(8, 8, F=f)
    assert layer.F is f


def test_F_and_explicit_subdepth_raises() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, 8, F=MonoLinear(8, 8, mode="mixed"), sub_depth=2)


def test_subdepth_below_one_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, 8, mode="mixed", sub_depth=0)


def _nondecreasing(layer: MonoResidual, in_f: int) -> None:
    torch.manual_seed(1)
    with torch.no_grad():  # exercise non-trivial gates (must hold for any params)
        layer.alpha.fill_(0.3)
        layer.beta.fill_(0.7)
    x = torch.randn(64, in_f, dtype=torch.float64)
    y0 = layer(x)
    for i in range(in_f):
        xp = x.clone()
        xp[:, i] += 0.5
        assert (layer(xp) - y0).min().item() >= -1e-9


def test_monotone_identity_skip() -> None:
    torch.manual_seed(0)
    _nondecreasing(
        MonoResidual(6, 6, mode="mixed", activation="elu", sub_depth=2).double(), 6
    )


def test_monotone_projection_skip() -> None:
    torch.manual_seed(0)
    _nondecreasing(
        MonoResidual(6, 4, mode="split", activation="elu", sub_depth=2).double(), 6
    )


def test_default_F_without_activation_raises() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="activation is required"):
        MonoResidual(8, 8, mode="mixed")


def test_F_and_activation_together_raises() -> None:  # noqa: N802
    f = MonoLinear(8, 8, mode="mixed")
    with pytest.raises(ValueError, match="either F or activation"):
        MonoResidual(8, 8, F=f, activation="elu")
