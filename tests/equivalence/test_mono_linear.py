# SPDX-License-Identifier: Apache-2.0
"""Cross-backend equivalence for the mono_linear kernel."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_linear")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch = pytest.importorskip("torch")
    from mononet.torch import _kernels as k

    dtype = torch.float64 if case.dtype == "float64" else torch.float32
    x = torch.tensor(case.array("x"), dtype=dtype)
    w = torch.tensor(case.array("weights"), dtype=dtype, requires_grad=True)
    b = torch.tensor(case.array("bias"), dtype=dtype, requires_grad=True)
    p = case.params
    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    y.sum().backward()  # type: ignore[no-untyped-call]
    assert w.grad is not None
    assert b.grad is not None
    return y.detach().numpy(), {"weights": w.grad.numpy(), "bias": b.grad.numpy()}


def _run_jax(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mononet.jax import _kernels as k

    p = case.params
    x = jnp.asarray(case.array("x"))
    w = jnp.asarray(case.array("weights"))
    b = jnp.asarray(case.array("bias"))

    def loss(w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return k.monotonic_dense(
            x, w, b, p["mode"], p["activation"], p["convex_fraction"]
        ).sum()

    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    gw, gb = jax.grad(loss, argnums=(0, 1))(w, b)
    return np.asarray(y), {"weights": np.asarray(gw), "bias": np.asarray(gb)}


def _run_keras(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pytest.importorskip("keras")
    import jax  # CI default KERAS_BACKEND=jax
    import jax.numpy as jnp

    from mononet.keras import _kernels as k

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    p = case.params
    x = jnp.asarray(case.array("x"))

    def loss(w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
        ).sum()

    w = jnp.asarray(case.array("weights"))
    b = jnp.asarray(case.array("bias"))
    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    gw, gb = jax.grad(loss, argnums=(0, 1))(w, b)
    return np.asarray(y), {"weights": np.asarray(gw), "bias": np.asarray(gb)}


_RUNNERS = {"torch": _run_torch, "jax": _run_jax, "keras": _run_keras}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_linear_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_linear runner for backend {BACKEND}")
    got, grads = runner(case)
    np.testing.assert_allclose(
        got, np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
    for key, expected in case.expected_grads.items():
        np.testing.assert_allclose(
            grads[key], np.asarray(expected), atol=1e-4, rtol=1e-3
        )
