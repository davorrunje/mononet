import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

pytest.importorskip("keras")

from tests.examples._loader import load_example


def test_risk_net_forward_and_monotone() -> None:
    """Keras RiskNet runs and is monotone in x_mono (dirs +1, +1, -1)."""
    mod = load_example("risk_net_keras.py")
    net = mod.RiskNet()
    rng = np.random.default_rng(0)
    x_mono = rng.standard_normal((16, 3)).astype("float32")
    x_free = rng.standard_normal((16, 2)).astype("float32")
    base = np.asarray(net(x_mono, x_free))
    assert base.shape == (16, 1)
    for j, sign in ((0, 1), (1, 1), (2, -1)):
        bumped = x_mono.copy()
        bumped[:, j] += 0.5
        diff = (np.asarray(net(bumped, x_free)) - base)[:, 0]
        assert (diff >= -1e-4).all() if sign > 0 else (diff <= 1e-4).all()
