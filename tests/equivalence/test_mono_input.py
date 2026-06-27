"""Cross-backend equivalence for the MonoInput sign-flip."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_input")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> np.ndarray:
    torch = pytest.importorskip("torch")
    from mononet.core.types import MonotonicityMask
    from mononet.torch import MonoInput

    directions = case.params["directions"]
    layer = MonoInput(MonotonicityMask(np.asarray(directions, dtype=np.int8)))
    out: np.ndarray = (
        layer(  # type: ignore[assignment]
            torch.tensor(case.array("x"), dtype=torch.float64)
        )
        .detach()
        .numpy()
    )
    return out


def _run_jax(case: EquivalenceCase) -> np.ndarray:
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from mononet.core.types import MonotonicityMask
    from mononet.jax import MonoInput

    layer = MonoInput(
        MonotonicityMask(np.asarray(case.params["directions"], dtype=np.int8))
    )
    return np.asarray(layer(jnp.asarray(case.array("x"))))


def _run_keras(case: EquivalenceCase) -> np.ndarray:
    pytest.importorskip("keras")
    import jax

    jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
    from keras import ops

    from mononet.core.types import MonotonicityMask
    from mononet.keras import MonoInput

    layer = MonoInput(
        MonotonicityMask(np.asarray(case.params["directions"], dtype=np.int8))
    )
    return np.asarray(layer(ops.convert_to_tensor(case.array("x"))))


_RUNNERS = {"torch": _run_torch, "jax": _run_jax, "keras": _run_keras}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_input_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_input runner for backend {BACKEND}")
    np.testing.assert_allclose(
        runner(case), np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
