# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the equivalence loader finds committed cases."""

from __future__ import annotations

import pytest

from tests.equivalence._cases import load_cases


@pytest.mark.parametrize("kind", ["mono_linear", "mono_residual", "mono_input"])
def test_cases_present(kind: str) -> None:
    cases = load_cases(kind)
    assert cases, f"no committed cases for {kind}"
