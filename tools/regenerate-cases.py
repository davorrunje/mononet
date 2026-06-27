# SPDX-License-Identifier: Apache-2.0
"""Regenerate committed equivalence vectors from the NumPy reference.

Run: `uv run python tools/regenerate-cases.py`. Vectors are the source of
truth for tests/equivalence; CI never regenerates them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mononet.core import numerics
from mononet.core import reference as ref
from mononet.core.types import ActivationSpec

if TYPE_CHECKING:
    from collections.abc import Callable

CASES = Path(__file__).resolve().parent.parent / "tests" / "equivalence" / "cases"
_FD_H = 1e-6
OUT_ATOL, OUT_RTOL = 1e-6, 1e-6


def _seed(name: str) -> int:
    """Deterministic per-case seed (stable across processes, unlike hash())."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _fd_grad(
    f: Callable[[np.ndarray], np.ndarray],  # type: ignore[type-arg]
    p: np.ndarray,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Central finite-difference gradient of `sum(f(p))` w.r.t. `p`."""
    g = np.zeros_like(p)
    flat = p.ravel()
    gflat = g.ravel()
    for i in range(flat.size):
        orig = float(flat[i])
        flat[i] = orig + _FD_H
        plus = float(f(p).sum())
        flat[i] = orig - _FD_H
        minus = float(f(p).sum())
        flat[i] = orig
        gflat[i] = (plus - minus) / (2 * _FD_H)
    return g


def _write(kind: str, name: str, payload: dict[str, Any]) -> None:
    d = CASES / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _dense_cases() -> None:
    grid = [
        ("1x1x1", 1, 1, 1, "switch", "relu", 0.5),
        ("4x2x3-switch-relu", 4, 2, 3, "switch", "relu", 0.5),
        ("8x7x12-switch-elu", 8, 7, 12, "switch", "elu", 0.5),
        ("4x2x3-abs-relu-c1", 4, 2, 3, "absolute", "relu", 1.0),
        ("4x2x3-abs-relu-c0", 4, 2, 3, "absolute", "relu", 0.0),
        ("3x5x11-abs-selu", 3, 5, 11, "absolute", "selu", 0.5),
        ("2x16x1-switch-softplus", 2, 16, 1, "switch", "softplus", 0.5),
    ]
    for name, b, n, m, mode, act, cf in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        w = rng.normal(size=(n, m))
        bias = rng.normal(size=m)
        spec = ActivationSpec(act)  # type: ignore[arg-type]
        out = ref.monotonic_dense(x, w, bias, mode, spec, cf)
        gw = _fd_grad(
            lambda ww, _x=x, _bias=bias, _mode=mode, _spec=spec, _cf=cf: (
                ref.monotonic_dense(_x, ww, _bias, _mode, _spec, _cf)
            ),
            w,
        )
        gb = _fd_grad(
            lambda bb, _x=x, _w=w, _mode=mode, _spec=spec, _cf=cf: ref.monotonic_dense(
                _x, _w, bb, _mode, _spec, _cf
            ),
            bias,
        )
        _write(
            "mono_linear",
            name,
            {
                "name": name,
                "inputs": {
                    "x": x.tolist(),
                    "weights": w.tolist(),
                    "bias": bias.tolist(),
                },
                "params": {
                    "mode": mode,
                    "activation": act,
                    "convex_fraction": cf,
                    "dtype": "float64",
                },
                "expected_output": out.tolist(),
                "expected_grads": {"weights": gw.tolist(), "bias": gb.tolist()},
                "atol": numerics.default_atol("float64"),
                "rtol": numerics.default_rtol("float64"),
            },
        )

    f32_variants = [
        ("2x16x1-switch-softplus", 2, 16, 1, "switch", "softplus", 0.5),
        ("8x7x12-switch-elu", 8, 7, 12, "switch", "elu", 0.5),
        ("3x5x11-abs-selu", 3, 5, 11, "absolute", "selu", 0.5),
    ]
    for orig_name, b, n, m, mode, act, cf in f32_variants:
        f32_name = f"{orig_name}-f32"
        rng = np.random.default_rng(_seed(orig_name))
        x = rng.normal(size=(b, n))
        w = rng.normal(size=(n, m))
        bias = rng.normal(size=m)
        spec = ActivationSpec(act)  # type: ignore[arg-type]
        out = ref.monotonic_dense(x, w, bias, mode, spec, cf)
        _write(
            "mono_linear",
            f32_name,
            {
                "name": f32_name,
                "inputs": {
                    "x": x.tolist(),
                    "weights": w.tolist(),
                    "bias": bias.tolist(),
                },
                "params": {
                    "mode": mode,
                    "activation": act,
                    "convex_fraction": cf,
                    "dtype": "float32",
                },
                "expected_output": out.tolist(),
                "expected_grads": {},
                "atol": numerics.default_atol("float32"),
                "rtol": numerics.default_rtol("float32"),
            },
        )


def _residual_cases() -> None:
    grid = [
        ("4x3x3-identity-switch", 4, 3, 3, None, "switch", "relu"),
        ("4x2x5-proj-switch", 4, 2, 5, (2, 5), "switch", "relu"),
        ("6x4x4-identity-abs", 6, 4, 4, None, "absolute", "elu"),
    ]
    for name, b, n, m, proj, mode, act in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        w = rng.normal(size=(n, m))
        bias = rng.normal(size=m)
        alpha = np.array(0.3)
        beta = np.array(0.5)
        skip = rng.normal(size=(n, m)) if proj else None
        spec = ActivationSpec(act)  # type: ignore[arg-type]

        def fwd(
            *,
            w: np.ndarray = w,  # type: ignore[type-arg]
            bias: np.ndarray = bias,  # type: ignore[type-arg]
            alpha: np.ndarray = alpha,  # type: ignore[type-arg]
            beta: np.ndarray = beta,  # type: ignore[type-arg]
            skip: np.ndarray | None = skip,  # type: ignore[type-arg]
            _x: np.ndarray = x,  # type: ignore[type-arg]
            _mode: str = mode,
            _spec: ActivationSpec = spec,
        ) -> np.ndarray:  # type: ignore[type-arg]
            return ref.monotonic_residual(
                _x,
                w,
                bias,
                alpha,
                beta,
                mode=_mode,
                activation=_spec,
                skip_weight=skip,
            )

        out = fwd()
        grads: dict[str, Any] = {
            "weights": _fd_grad(lambda v: fwd(w=v), w).tolist(),
            "bias": _fd_grad(lambda v: fwd(bias=v), bias).tolist(),
            "alpha": _fd_grad(lambda v: fwd(alpha=v), alpha).tolist(),
            "beta": _fd_grad(lambda v: fwd(beta=v), beta).tolist(),
        }
        inputs: dict[str, Any] = {
            "x": x.tolist(),
            "weights": w.tolist(),
            "bias": bias.tolist(),
            "alpha": alpha.tolist(),
            "beta": beta.tolist(),
        }
        if skip is not None:
            inputs["skip_weight"] = skip.tolist()
            grads["skip_weight"] = _fd_grad(lambda v: fwd(skip=v), skip).tolist()
        _write(
            "mono_residual",
            name,
            {
                "name": name,
                "inputs": inputs,
                "params": {
                    "mode": mode,
                    "activation": act,
                    "convex_fraction": 0.5,
                    "alpha_gate": "shifted_elu",
                    "beta_gate": "scaled_elu",
                    "has_projection": skip is not None,
                    "dtype": "float64",
                },
                "expected_output": out.tolist(),
                "expected_grads": grads,
                "atol": OUT_ATOL,
                "rtol": OUT_RTOL,
            },
        )


def _input_cases() -> None:
    grid = [
        ("scalar-plus", 4, 3, [1, 1, 1]),
        ("scalar-minus", 4, 3, [-1, -1, -1]),
        ("mixed", 5, 4, [1, -1, 1, -1]),
    ]
    for name, b, n, directions in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        d = np.asarray(directions, dtype=np.float64)
        out = x * d
        _write(
            "mono_input",
            name,
            {
                "name": name,
                "inputs": {"x": x.tolist()},
                "params": {"directions": directions, "dtype": "float64"},
                "expected_output": out.tolist(),
                "expected_grads": {},
                "atol": OUT_ATOL,
                "rtol": OUT_RTOL,
            },
        )


def main() -> None:
    """Regenerate all case files and write a reference git hash."""
    _dense_cases()
    _residual_cases()
    _input_cases()
    sha = subprocess.run(
        ["git", "hash-object", "mononet/core/reference.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (CASES / "REFERENCE_HASH").write_text(sha + "\n")
    print(f"Written REFERENCE_HASH: {sha}")
    print("Done.")


if __name__ == "__main__":
    main()
