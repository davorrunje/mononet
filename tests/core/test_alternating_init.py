import math

import numpy as np
import pytest

from mononet.core.init import (
    _expect,
    _solve_gain,
    alternating_init_params,
    alternating_weight_bias,
)


def test_entry_layer_is_unit_gain_zero_bias() -> None:
    # m_in=0 (entry): s=1 so gain == G (the mixed unit-variance gain); bias == 0.
    gain, _out_mean = alternating_init_params("relu", m_in=0.0, convex=True)
    assert gain == pytest.approx(_solve_gain("relu", 0.0))
    ws, bias = alternating_weight_bias(gain, m_in=0.0, fan_in=32)
    assert bias == 0.0
    assert ws == pytest.approx(gain / math.sqrt(32))


def test_out_mean_is_signed_per_activation_constant() -> None:
    g_unit = _solve_gain("relu", 0.0)
    expected = _expect("relu", 0.0, g_unit, moment=1)  # E[relu(G·H)]
    _conv_gain, conv_mean = alternating_init_params("relu", m_in=-0.4, convex=True)
    _, conc_mean = alternating_init_params("relu", m_in=0.4, convex=False)
    assert conv_mean == pytest.approx(expected)
    assert conc_mean == pytest.approx(-expected)


def test_interior_gain_shrinks_and_bias_centers() -> None:
    # interior layer fed by opposite class (m_in != 0): gain = G/s < G, bias sign
    # opposes m_in (pulls the |W|-inflated preactivation back to 0).
    g_unit = _solve_gain("elu", 0.0)
    m_in = 0.6
    gain, _ = alternating_init_params("elu", m_in=m_in, convex=False)
    s = math.sqrt(1.0 + m_in**2 * (1.0 - 2.0 / math.pi))
    assert gain == pytest.approx(g_unit / s)
    _, bias = alternating_weight_bias(gain, m_in=m_in, fan_in=16)
    assert bias < 0.0  # m_in > 0 -> negative centering bias


def test_montecarlo_output_variance_is_unit() -> None:
    # A pure layer built with these params has ~unit output variance under
    # a standard-normal-ish |W|·h + b, confirming the centering/gain.
    rng = np.random.default_rng(0)
    fan = 64
    m_in = 0.3
    gain, _ = alternating_init_params("relu", m_in=m_in, convex=True)
    ws, bias = alternating_weight_bias(gain, m_in, fan)
    W = np.abs(rng.normal(0.0, ws, size=(fan, 256)))  # |W|
    h = rng.normal(m_in, 1.0, size=(4096, fan))  # input mean m_in, unit var
    z = h @ W + bias
    out = np.maximum(0.0, z)  # relu (convex)
    assert out.var() == pytest.approx(1.0, abs=0.15)


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        alternating_init_params("gelu", m_in=0.0, convex=True)
