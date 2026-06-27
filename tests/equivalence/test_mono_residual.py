"""Cross-backend equivalence for the mono_residual kernel."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_residual")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch = pytest.importorskip("torch")
    from mononet.torch import _kernels as k

    p = case.params
    x = torch.tensor(case.array("x"), dtype=torch.float64)
    w = torch.tensor(case.array("weights"), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(case.array("bias"), dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(case.array("alpha"), dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(case.array("beta"), dtype=torch.float64, requires_grad=True)
    sw = (
        torch.tensor(case.array("skip_weight"), dtype=torch.float64, requires_grad=True)
        if p["has_projection"]
        else None
    )
    y = k.monotonic_residual(
        x,
        w,
        b,
        alpha,
        beta,
        mode=p["mode"],
        activation_name=p["activation"],
        convex_fraction=p["convex_fraction"],
        alpha_gate=p["alpha_gate"],
        beta_gate=p["beta_gate"],
        skip_weight=sw,
    )
    y.sum().backward()  # type: ignore[no-untyped-call]
    assert w.grad is not None
    assert b.grad is not None
    assert alpha.grad is not None
    assert beta.grad is not None
    grads: dict[str, np.ndarray] = {
        "weights": w.grad.numpy(),
        "bias": b.grad.numpy(),
        "alpha": alpha.grad.numpy(),
        "beta": beta.grad.numpy(),
    }
    if sw is not None:
        assert sw.grad is not None
        grads["skip_weight"] = sw.grad.numpy()
    return y.detach().numpy(), grads


def _run_jax(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from mononet.jax import _kernels as k

    p = case.params
    x = jnp.asarray(case.array("x"))
    args = {n: jnp.asarray(case.array(n)) for n in ("weights", "bias", "alpha", "beta")}
    if p["has_projection"]:
        args["skip_weight"] = jnp.asarray(case.array("skip_weight"))
    names = list(args)

    def fwd(*vals: jnp.ndarray) -> jnp.ndarray:
        kw = dict(zip(names, vals, strict=True))
        sw = kw.pop("skip_weight", None)
        return k.monotonic_residual(
            x,
            kw["weights"],
            kw["bias"],
            kw["alpha"],
            kw["beta"],
            mode=p["mode"],
            activation_name=p["activation"],
            convex_fraction=p["convex_fraction"],
            alpha_gate=p["alpha_gate"],
            beta_gate=p["beta_gate"],
            skip_weight=sw,
        )

    y = fwd(*args.values())
    grads = jax.grad(lambda *v: fwd(*v).sum(), argnums=tuple(range(len(names))))(
        *args.values()
    )
    return np.asarray(y), {n: np.asarray(g) for n, g in zip(names, grads, strict=True)}


_RUNNERS = {"torch": _run_torch, "jax": _run_jax}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_residual_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_residual runner for backend {BACKEND}")
    got, grads = runner(case)
    np.testing.assert_allclose(
        got, np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
    for key, expected in case.expected_grads.items():
        np.testing.assert_allclose(
            grads[key], np.asarray(expected), atol=1e-4, rtol=1e-3
        )
