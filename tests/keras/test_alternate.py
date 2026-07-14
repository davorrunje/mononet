# SPDX-License-Identifier: Apache-2.0
"""Keras 3 tests for mode="alternate" + init-time prev=."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")

from typing import TYPE_CHECKING, Any  # noqa: E402

from mononet.keras import MonoDense  # noqa: E402

if TYPE_CHECKING:
    from mononet.core.types import ActivationName


def _stack(
    act: ActivationName = "relu", depth: int = 4, d: int = 4, h: int = 16
) -> Any:
    keras.utils.set_random_seed(0)
    layers: list[MonoDense] = []
    prev: MonoDense | None = None
    for _ in range(depth):
        lay = MonoDense(h, mode="alternate", activation=act, prev=prev)
        layers.append(lay)
        prev = lay
    layers.append(MonoDense(1, mode="mixed", activation="identity"))
    model = keras.Sequential(layers)
    model(np.zeros((1, d), dtype="float32"))  # trigger build for every layer
    return model


def _alternate_layers(net: Any) -> list[MonoDense]:
    return [m for m in net.layers if isinstance(m, MonoDense) and m.mode == "alternate"]


def test_prev_alternates_phase_and_entry_is_convex() -> None:
    net = _stack()
    alt = _alternate_layers(net)
    assert [m._alt_convex for m in alt] == [True, False, True, False]


def test_entry_bias_zero_interior_bias_alternates_sign() -> None:
    net = _stack()
    alt = _alternate_layers(net)
    assert alt[0].b is not None
    assert alt[1].b is not None
    assert alt[2].b is not None
    assert float(np.max(np.abs(np.asarray(alt[0].b)))) == pytest.approx(0.0, abs=1e-6)
    assert float(np.mean(np.asarray(alt[1].b))) < 0.0  # concave interior
    assert float(np.mean(np.asarray(alt[2].b))) > 0.0  # convex interior


def test_prev_not_retained() -> None:
    net = _stack()
    alt = _alternate_layers(net)
    assert all("prev" not in vars(m) for m in alt)


def test_alternate_is_monotone_nondecreasing() -> None:
    net = _stack()
    x = np.zeros((1, 4), dtype="float32")
    base = np.asarray(net(x))
    for j in range(4):
        bumped = x.copy()
        bumped[0, j] += 1e-2
        assert float((np.asarray(net(bumped)) - base).item()) >= -1e-5


def test_convex_fraction_rejected_for_alternate() -> None:
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoDense(8, mode="alternate", activation="relu", convex_fraction=0.3)


def test_prev_rejected_for_non_alternate() -> None:
    entry = MonoDense(8, mode="alternate", activation="relu")
    with pytest.raises(ValueError, match="prev"):
        MonoDense(8, mode="mixed", activation="relu", prev=entry)


def test_prev_must_be_alternate() -> None:
    mixed = MonoDense(8, mode="mixed", activation="relu")
    with pytest.raises(ValueError, match="alternate"):
        MonoDense(8, mode="alternate", activation="relu", prev=mixed)


def test_deep_alternate_trains_stably() -> None:
    # depth-8 plain alternate stack does not diverge (contrast: mixed diverges).
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (2000, 4)).astype("float32")
    y = (1 / (1 + np.exp(-3 * (x - 0.1)))).sum(1, keepdims=True).astype("float32")
    y = (y - y.mean()) / y.std()
    net = _stack(depth=8)
    net.compile(optimizer=keras.optimizers.Adam(1e-2), loss="mse")
    net.fit(x, y, epochs=300, batch_size=2000, verbose=0)
    loss = float(net.evaluate(x, y, verbose=0))
    assert loss < 0.9  # beats predict-the-mean (~1.0)


def test_mono_residual_rejects_alternate() -> None:
    from mononet.keras import MonoResidual

    with pytest.raises(ValueError, match="alternate"):
        MonoResidual(8, mode="alternate", activation="relu")
