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

# Expected committed golden-case counts. This is a hard assertion, not a
# skip: if the goldens file is deleted, emptied, or truncated, the fidelity
# anchor (test_monodense_matches_golden / test_builder_matches_golden) would
# otherwise vanish from the parametrized suite with green CI. A regenerated
# goldens file with a different intentional case count must update these
# constants alongside it.
_EXPECTED_MONODENSE_CASES = 4
_EXPECTED_BUILDER_CASES = 2


def test_goldens_present() -> None:
    """Fail (not skip) if the committed golden cases are missing or truncated."""
    assert len(_MONODENSE_CASES) == _EXPECTED_MONODENSE_CASES, (
        f"expected {_EXPECTED_MONODENSE_CASES} monodense golden cases, "
        f"found {len(_MONODENSE_CASES)}; run tools/gen-legacy-goldens.py"
    )
    assert len(_BUILDER_CASES) == _EXPECTED_BUILDER_CASES, (
        f"expected {_EXPECTED_BUILDER_CASES} builder golden cases, "
        f"found {len(_BUILDER_CASES)}; run tools/gen-legacy-goldens.py"
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


def _match_layer_weights(
    expected_shapes: list[tuple[int, ...]],
    slice_weights: list[np.ndarray],
    layer_name: str,
) -> list[np.ndarray]:
    """Reorder one layer's weight slice to match its expected shape sequence.

    Matches each expected shape to the unique array of that shape still
    available in ``slice_weights``. Within a Dense/MonoDense layer the
    kernel (2-D) and bias (1-D) have distinct shapes, so the match is
    unambiguous.

    :param expected_shapes: The layer's own weight shapes, in its own order
        (``[w.shape for w in layer.weights]``).
    :param slice_weights: The golden arrays recorded for this layer, in
        (possibly wrong) order.
    :param layer_name: Layer name, used only for the error message.
    :returns: ``slice_weights`` reordered to match ``expected_shapes``.
    :raises AssertionError: If some expected shape has zero or more than one
        match among the remaining candidates -- a real structural mismatch
        that must not be silently guessed at.
    """
    available = list(range(len(slice_weights)))
    reordered: list[np.ndarray] = []
    for shape in expected_shapes:
        matches = [i for i in available if tuple(slice_weights[i].shape) == shape]
        if len(matches) != 1:
            candidate_shapes = [tuple(slice_weights[i].shape) for i in available]
            raise AssertionError(
                f"Layer {layer_name!r}: expected exactly one weight of shape "
                f"{shape} among {candidate_shapes}, found {len(matches)}."
            )
        chosen = matches[0]
        reordered.append(slice_weights[chosen])
        available.remove(chosen)
    return reordered


def _canonicalize_dense_weight_order(
    model: Any,
    golden_weights: list[np.ndarray],
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
    always ``[kernel, bias]``.

    ``golden_weights`` is a *flat* list spanning the whole model, so the
    reorder must not cross layer boundaries: two adjacent layers of the same
    width could otherwise be shape-matched into the wrong permutation. This
    helper instead walks the already-built port ``model``'s weight-bearing
    layers in order (the golden and the port share the same functional
    graph, so the layer order and per-layer weight count line up) and
    reorders each layer's own slice of the flat list to match that layer's
    expected shapes via :func:`_match_layer_weights`. The reorder is thus
    always scoped to a single layer.

    :param model: The already-built port Keras model; its ``layers`` and
        each layer's ``weights`` supply the target per-layer shape order.
    :param golden_weights: Flat weight list as recorded by the original's
        ``model.get_weights()``.
    :returns: ``golden_weights`` reordered to match ``model.get_weights()``'s
        expected layout.
    :raises AssertionError: If the flat golden list's length does not match
        the sum of the port model's weight-bearing layers' weight counts.
    """
    weight_layers = [layer for layer in model.layers if len(layer.weights) > 0]
    out: list[np.ndarray] = []
    i = 0
    for layer in weight_layers:
        n = len(layer.weights)
        chunk = golden_weights[i : i + n]
        i += n
        expected_shapes = [tuple(w.shape) for w in layer.weights]
        out.extend(_match_layer_weights(expected_shapes, chunk, layer.name))
    if i != len(golden_weights):
        raise AssertionError(
            f"Golden weight count {len(golden_weights)} does not match the "
            f"port model's weight-bearing layer total {i}."
        )
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
        model, [np.array(w, dtype="float32") for w in case["weights"]]
    )
    model.set_weights(golden_weights)
    x = np.array(case["input"], dtype="float32")
    feed = [keras.ops.convert_to_tensor(x[:, i : i + 1]) for i in range(n)]
    got = np.asarray(model(feed))
    assert np.allclose(got, np.array(case["output"]), atol=1e-5, rtol=1e-5)
