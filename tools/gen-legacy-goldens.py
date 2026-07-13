# SPDX-License-Identifier: Apache-2.0
r"""One-time generator for legacy MonoDense golden vectors.

Run manually in an ephemeral environment with the ORIGINAL TensorFlow package —
this is NOT run in CI:

    uv run --python 3.11 --with 'monotonic-nn==0.3.5' --with tensorflow \\
        python tools/gen-legacy-goldens.py

It imports the original ``airt`` implementation, runs a fixed battery of cases,
and writes JSON goldens under ``tests/legacy/goldens/``. The ported layer is
asserted equal to these vectors in ``tests/legacy/test_equivalence.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from airt.keras.layers import MonoDense  # original TF implementation

GOLDENS = Path(__file__).resolve().parent.parent / "tests" / "legacy" / "goldens"


def _monodense_cases() -> list[dict]:
    rng = np.random.default_rng(0)
    specs = [
        {
            "name": "relu_inc_bias",
            "units": 4,
            "activation": "relu",
            "monotonicity_indicator": [1, 1, -1, 0],
            "use_bias": True,
            "activation_weights": (7.0, 7.0, 2.0),
            "in_f": 4,
        },
        {
            "name": "elu_convex",
            "units": 6,
            "activation": "elu",
            "monotonicity_indicator": 1,
            "is_convex": True,
            "use_bias": True,
            "activation_weights": (7.0, 7.0, 2.0),
            "in_f": 3,
        },
        {
            "name": "elu_concave_nobias",
            "units": 6,
            "activation": "elu",
            "monotonicity_indicator": 1,
            "is_concave": True,
            "use_bias": False,
            "activation_weights": (7.0, 7.0, 2.0),
            "in_f": 3,
        },
        {
            "name": "relu_custom_weights",
            "units": 10,
            "activation": "relu",
            "monotonicity_indicator": [1, -1, 0],
            "use_bias": True,
            "activation_weights": (2.0, 3.0, 1.0),
            "in_f": 3,
        },
    ]
    out = []
    for s in specs:
        layer = MonoDense(
            s["units"],
            activation=s["activation"],
            monotonicity_indicator=s["monotonicity_indicator"],
            is_convex=s.get("is_convex", False),
            is_concave=s.get("is_concave", False),
            activation_weights=s["activation_weights"],
            use_bias=s["use_bias"],
        )
        layer.build((None, s["in_f"]))
        kernel = rng.standard_normal((s["in_f"], s["units"])).astype("float32")
        weights = [kernel]
        if s["use_bias"]:
            bias = rng.standard_normal((s["units"],)).astype("float32")
            weights.append(bias)
        layer.set_weights(weights)
        x = rng.standard_normal((5, s["in_f"])).astype("float32")
        y = np.asarray(layer(tf.convert_to_tensor(x)))
        case = {
            "name": s["name"],
            "units": s["units"],
            "activation": s["activation"],
            "monotonicity_indicator": s["monotonicity_indicator"],
            "is_convex": s.get("is_convex", False),
            "is_concave": s.get("is_concave", False),
            "activation_weights": list(s["activation_weights"]),
            "use_bias": s["use_bias"],
            "kernel": kernel.tolist(),
            "input": x.tolist(),
            "output": y.tolist(),
        }
        if s["use_bias"]:
            case["bias"] = weights[1].tolist()
        out.append(case)
    return out


def _builder_cases() -> list[dict]:
    # The original builders take a list/dict of per-feature single-column
    # Input tensors (their real usage) — NOT a single multi-feature tensor.
    from airt.keras.layers import MonoDense as M

    rng = np.random.default_rng(1)
    out = []
    specs = [
        {
            "name": "type1_basic",
            "builder": "type_1",
            "n_features": 4,
            "kwargs": {
                "units": 8,
                "final_units": 1,
                "activation": "elu",
                "n_layers": 3,
                "monotonicity_indicator": [1, 1, -1, 0],
            },
        },
        {
            "name": "type2_basic",
            "builder": "type_2",
            "n_features": 4,
            "kwargs": {
                "units": 8,
                "final_units": 1,
                "activation": "elu",
                "n_layers": 2,
                "monotonicity_indicator": [1, -1, 0, 1],
            },
        },
    ]
    for s in specs:
        n = s["n_features"]
        inputs = [tf.keras.Input(shape=(1,)) for _ in range(n)]
        build = M.create_type_1 if s["builder"] == "type_1" else M.create_type_2
        y = build(inputs, **s["kwargs"])
        model = tf.keras.Model(inputs, y)
        weights = [w.tolist() for w in model.get_weights()]
        x = rng.standard_normal((5, n)).astype("float32")
        feed = [tf.convert_to_tensor(x[:, i : i + 1]) for i in range(n)]
        out.append(
            {
                "name": s["name"],
                "builder": s["builder"],
                "kwargs": s["kwargs"],
                "n_features": n,
                "weights": weights,
                "input": x.tolist(),
                "output": np.asarray(model(feed)).tolist(),
            }
        )
    return out


def main() -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    (GOLDENS / "monodense_cases.json").write_text(
        json.dumps(_monodense_cases(), indent=2)
    )
    (GOLDENS / "builder_cases.json").write_text(json.dumps(_builder_cases(), indent=2))
    print(f"wrote goldens to {GOLDENS}")


if __name__ == "__main__":
    main()
