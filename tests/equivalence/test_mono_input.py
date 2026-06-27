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
    out: np.ndarray = layer(  # type: ignore[assignment]
        torch.tensor(case.array("x"), dtype=torch.float64)
    ).detach().numpy()
    return out


_RUNNERS = {"torch": _run_torch}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_input_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_input runner for backend {BACKEND}")
    np.testing.assert_allclose(
        runner(case), np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
