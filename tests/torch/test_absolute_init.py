import math

import pytest

torch = pytest.importorskip("torch")

from mononet.core.init import absolute_init_params  # noqa: E402
from mononet.torch import MonoLinear  # noqa: E402


def test_absolute_default_weight_scale_and_bias() -> None:
    torch.manual_seed(0)
    in_f, units = 256, 512
    layer = MonoLinear(in_f, units, mode="absolute", activation="elu")
    gain, bias = absolute_init_params("elu", 0.5)
    got = float(layer.weight.detach().std())
    assert abs(got - gain / math.sqrt(in_f)) < 0.05 * gain / math.sqrt(in_f)
    assert layer.bias is not None
    assert torch.allclose(
        layer.bias.detach(), torch.full((units,), bias, dtype=layer.bias.dtype)
    )


def test_absolute_bias_nonzero_off_half() -> None:
    torch.manual_seed(0)
    layer = MonoLinear(64, 64, mode="absolute", activation="elu", convex_fraction=0.25)
    _, bias = absolute_init_params("elu", 0.25)
    assert bias != 0.0
    assert layer.bias is not None
    assert torch.allclose(
        layer.bias.detach(), torch.full((64,), bias, dtype=layer.bias.dtype)
    )


def test_explicit_init_overrides_absolute() -> None:
    torch.manual_seed(0)
    layer = MonoLinear(64, 64, mode="absolute", activation="elu", init="he_normal")
    assert layer.bias is not None
    assert torch.allclose(layer.bias.detach(), torch.zeros(64, dtype=layer.bias.dtype))


def test_switch_default_unchanged() -> None:
    torch.manual_seed(0)
    layer = MonoLinear(64, 64, mode="switch", activation="elu")
    assert layer.bias is not None
    assert torch.allclose(layer.bias.detach(), torch.zeros(64, dtype=layer.bias.dtype))
