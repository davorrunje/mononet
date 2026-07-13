# SPDX-License-Identifier: Apache-2.0
"""Assert the ported layer matches committed goldens from the original impl."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense, create_type_1, create_type_2  # noqa: E402

GOLDENS = Path(__file__).parent / "goldens"


def _load(name: str) -> list[dict[str, Any]]:
    path = GOLDENS / name
    if not path.exists():
        return []
    return cast("list[dict[str, Any]]", json.loads(path.read_text()))


_MONODENSE_CASES = _load("monodense_cases.json")
_BUILDER_CASES = _load("builder_cases.json")

pytestmark = pytest.mark.skipif(
    not _MONODENSE_CASES and not _BUILDER_CASES,
    reason="legacy goldens not generated; run tools/gen-legacy-goldens.py",
)


@pytest.mark.parametrize("case", _MONODENSE_CASES, ids=lambda c: c["name"])
def test_monodense_matches_golden(case: dict[str, Any]) -> None:
    legacy._WARNED = True
    layer = MonoDense(
        case["units"],
        activation=case["activation"],
        monotonicity_indicator=case["monotonicity_indicator"],
        is_convex=case["is_convex"],
        is_concave=case["is_concave"],
        activation_weights=tuple(case["activation_weights"]),
        use_bias=case["use_bias"],
    )
    x = np.array(case["input"], dtype="float32")
    layer.build((None, x.shape[-1]))
    weights = [np.array(case["kernel"], dtype="float32")]
    if case["use_bias"]:
        weights.append(np.array(case["bias"], dtype="float32"))
    layer.set_weights(weights)
    got = np.asarray(layer(keras.ops.convert_to_tensor(x)))
    assert np.allclose(got, np.array(case["output"]), atol=1e-5, rtol=1e-5)


def _canonicalize_dense_weight_order(
    weights: list[np.ndarray],
) -> list[np.ndarray]:
    """Undo the original TF impl's ``[bias, kernel]`` MonoDense quirk.

    The original ``airt`` ``MonoDense.call`` temporarily reassigns
    ``self.kernel`` to a masked tensor via a context manager while tracing a
    Functional model. That reassignment makes TF's checkpoint-dependency
    tracking re-register the ``kernel`` attribute *after* ``bias``, so
    ``model.get_weights()`` yields ``[bias, kernel]`` for every ``MonoDense``
    layer in the graph (verified against ``monotonic-nn==0.3.5`` directly:
    a standalone, un-called layer keeps ``[kernel, bias]``; a plain
    ``keras.layers.Dense`` in the same graph is unaffected). This port's
    ``call`` never reassigns ``self.kernel``, so its own weight order is
    always ``[kernel, bias]``. This helper detects the swapped
    ``(1-D bias, 2-D kernel)`` adjacent pairs by shape and restores the
    canonical order before ``set_weights`` -- it corrects a bookkeeping
    artifact of the reference implementation, not a numerical discrepancy.

    :param weights: Flat weight list as recorded by the original's
        ``model.get_weights()``.
    :returns: The same arrays with any swapped ``(bias, kernel)`` pairs
        reordered to ``(kernel, bias)``.
    """
    out = []
    i = 0
    while i < len(weights):
        w = weights[i]
        nxt = weights[i + 1] if i + 1 < len(weights) else None
        if (
            nxt is not None
            and w.ndim == 1
            and nxt.ndim == 2
            and nxt.shape[-1] == w.shape[0]
        ):
            out.append(nxt)
            out.append(w)
            i += 2
        else:
            out.append(w)
            i += 1
    return out


@pytest.mark.parametrize("case", _BUILDER_CASES, ids=lambda c: c["name"])
def test_builder_matches_golden(case: dict[str, Any]) -> None:
    legacy._WARNED = True
    n = case["n_features"]
    inputs = [keras.Input(shape=(1,)) for _ in range(n)]
    build = create_type_1 if case["builder"] == "type_1" else create_type_2
    out = build(inputs, **case["kwargs"])
    model = keras.Model(inputs, out)
    golden_weights = _canonicalize_dense_weight_order(
        [np.array(w, dtype="float32") for w in case["weights"]]
    )
    model.set_weights(golden_weights)
    x = np.array(case["input"], dtype="float32")
    feed = [keras.ops.convert_to_tensor(x[:, i : i + 1]) for i in range(n)]
    got = np.asarray(model(feed))
    assert np.allclose(got, np.array(case["output"]), atol=1e-5, rtol=1e-5)
