from __future__ import annotations

from typing import Any

import pytest

from benchmarks._common.size_ladder_report import delta_by_n


def _rec(n: int, arm: str, iqm: float, values: list[float]) -> dict[str, Any]:
    return {"n": n, "arm": arm, "test_iqm": iqm, "test_values": values}


def test_delta_by_n_pairs_arms_and_signs_delta() -> None:
    """Delta = deep_iqm - shallow_iqm per N, with a band bracketing it."""
    records = [
        _rec(100, "shallow", 0.60, [0.59, 0.61]),
        _rec(100, "deep", 0.58, [0.57, 0.59]),
        _rec(400, "shallow", 0.60, [0.60, 0.60]),
        _rec(400, "deep", 0.66, [0.65, 0.67]),
    ]
    rows = delta_by_n(records)
    by_n = {r["n"]: r for r in rows}
    assert by_n[100]["delta"] == pytest.approx(-0.02, abs=1e-9)
    assert by_n[400]["delta"] == pytest.approx(0.06, abs=1e-9)
    assert by_n[400]["delta_lo"] <= by_n[400]["delta"] <= by_n[400]["delta_hi"]
