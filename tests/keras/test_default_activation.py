import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

pytest.importorskip("keras")

from mononet.keras import MonoDense


def test_default_activation_is_affine() -> None:
    layer = MonoDense(8, mode="split")
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((5, 4)).astype("float32")
    x2 = rng.standard_normal((5, 4)).astype("float32")
    mid = np.asarray(layer((x1 + x2) / 2))
    avg = (np.asarray(layer(x1)) + np.asarray(layer(x2))) / 2
    np.testing.assert_allclose(mid, avg, rtol=1e-5, atol=1e-5)
