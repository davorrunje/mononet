import pytest

from mononet.core.init import _expect, absolute_init_params  # helpers used below

ACTS = ["relu", "elu", "selu", "softplus"]


@pytest.mark.parametrize("act", ACTS)
def test_gain_preserves_output_variance(act: str) -> None:
    gain, bias = absolute_init_params(act, 0.5)
    # pre-activation ~ N(bias, gain^2); output variance must be ~1
    var = _expect(act, bias, gain, moment=2) - _expect(act, bias, gain, moment=1) ** 2
    assert gain > 0.0
    assert abs(var - 1.0) < 1e-2


@pytest.mark.parametrize("act", ACTS)
def test_bias_zero_at_half(act: str) -> None:
    _, bias = absolute_init_params(act, 0.5)
    assert bias == 0.0


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("f", [0.25, 0.75])
def test_layer_mean_zero_off_half(act: str, f: float) -> None:
    gain, bias = absolute_init_params(act, f)
    # layer mean = f*E[act(H+b)] - (1-f)*E[act(-(H+b))], H~N(0,1)
    conv = _expect(act, bias, gain, moment=1)
    conc = _expect(act, -bias, gain, moment=1)  # E[act(-H - b)] via H symmetry
    layer_mean = f * conv - (1.0 - f) * conc
    assert abs(layer_mean) < 1e-2


def test_deterministic() -> None:
    assert absolute_init_params("elu", 0.5) == absolute_init_params("elu", 0.5)


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        absolute_init_params("gelu", 0.5)


def test_identity_activation_is_supported() -> None:
    # Exercises the identity branch of the private `_act` helper: for
    # identity, act(h) = h, so Var[gain*H] = 1 at gain == 1 and bias == 0.
    gain, bias = absolute_init_params("identity", 0.5)
    assert gain == pytest.approx(1.0)
    assert bias == 0.0


def test_bisect_returns_midpoint_when_iterations_exhausted() -> None:
    from mononet.core.init import _bisect

    # Root of (x - 1/3) is 1/3; with tol=0 the |fmid| < tol test never fires,
    # so after `iters` steps the loop falls through to `return 0.5*(lo+hi)`.
    root = _bisect(lambda x: x - 1.0 / 3.0, 0.0, 1.0, tol=0.0, iters=5)
    assert abs(root - 1.0 / 3.0) < 0.1
