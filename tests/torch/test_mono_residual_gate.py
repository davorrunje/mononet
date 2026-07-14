"""Near-zero init of the default F, and gate defaults, for MonoResidual (torch)."""

import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear, MonoResidual  # noqa: E402


def _last_linear(block: MonoResidual) -> MonoLinear:
    f = block.F
    if isinstance(f, MonoLinear):
        return f
    assert isinstance(f, torch.nn.Sequential)
    last = list(f)[-1]
    assert isinstance(last, MonoLinear)
    return last


def test_default_F_last_layer_is_near_zero_but_nonzero() -> None:  # noqa: N802
    torch.manual_seed(0)
    block = MonoResidual(32, 32, mode="mixed", activation="elu")
    last = _last_linear(block)
    wnorm = float(last.weight.detach().abs().sum())
    # small but NOT exactly zero (exact zero would freeze under |W|)
    assert wnorm > 0.0
    assert wnorm < 1.0  # heavily attenuated vs a normal init (~tens)
    # bias zeroed
    assert last.bias is not None
    assert float(last.bias.detach().abs().sum()) == 0.0


def test_default_block_is_near_identity_at_init() -> None:
    torch.manual_seed(0)
    block = MonoResidual(32, 32, mode="mixed", activation="elu")
    x = torch.randn(8, 32)
    fx_rms = float(block.F(x).detach().pow(2).mean().sqrt())
    assert fx_rms < 0.2  # F(x) ~= 0 at init => block ~= g_alpha * skip


def test_custom_F_is_not_near_zeroed() -> None:  # noqa: N802
    torch.manual_seed(0)
    custom = MonoLinear(32, 32, mode="mixed", activation="elu")
    before = float(custom.weight.detach().abs().sum())
    block = MonoResidual(32, 32, F=custom)
    assert isinstance(block.F, MonoLinear)
    after = float(block.F.weight.detach().abs().sum())
    assert after == before  # untouched


def test_near_zero_scale_is_user_tunable() -> None:
    torch.manual_seed(0)
    small = _last_linear(MonoResidual(32, 32, mode="mixed", activation="elu"))
    torch.manual_seed(0)
    big = _last_linear(
        MonoResidual(32, 32, mode="mixed", activation="elu", near_zero_scale=2e-3)
    )
    # same seed => 2e-3 gives ~2x the weight magnitude of the 1e-3 default
    ratio = float(big.weight.detach().abs().sum()) / float(
        small.weight.detach().abs().sum()
    )
    assert ratio == pytest.approx(2.0, rel=1e-5)
    # 0.0 reproduces exact-zero
    torch.manual_seed(0)
    zero = _last_linear(
        MonoResidual(32, 32, mode="mixed", activation="elu", near_zero_scale=0.0)
    )
    assert float(zero.weight.detach().abs().sum()) == 0.0


def test_near_zero_scale_with_bias_false() -> None:
    # covers the no-bias branch of near-zero init: weight scaled, no bias to zero
    layer = MonoLinear(
        4, 4, mode="mixed", activation="elu", bias=False, near_zero_scale=1e-3
    )
    assert layer.bias is None
    assert float(layer.weight.detach().abs().sum()) > 0.0  # scaled but nonzero
    layer(torch.zeros(2, 4))  # forward runs without a bias
