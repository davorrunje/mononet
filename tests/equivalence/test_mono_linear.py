"""Cross-backend equivalence for the mono_linear kernel."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_linear")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch = pytest.importorskip("torch")
    from mononet.torch import _kernels as k

    x = torch.tensor(case.array("x"), dtype=torch.float64)
    w = torch.tensor(case.array("weights"), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(case.array("bias"), dtype=torch.float64, requires_grad=True)
    p = case.params
    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    y.sum().backward()  # type: ignore[no-untyped-call]
    assert w.grad is not None
    assert b.grad is not None
    return y.detach().numpy(), {"weights": w.grad.numpy(), "bias": b.grad.numpy()}


_RUNNERS = {"torch": _run_torch}  # jax/keras runners added in their phases


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
