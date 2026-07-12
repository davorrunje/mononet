from __future__ import annotations

import pytest

from benchmarks._common.screen_gate import DEFAULT_MARGIN, gate


@pytest.mark.parametrize(
    ("lo", "point", "expect"),
    [
        (0.001, 0.010, "ladder"),  # CI clears 0 and point clears margin
        (0.001, 0.004, "standard"),  # CI clears 0 but point below margin
        (-0.001, 0.010, "standard"),  # point big but CI touches 0
        (0.0, 0.010, "standard"),  # lo == 0 is not > 0
        (0.006, 0.005, "ladder"),  # point == margin qualifies
    ],
)
def test_gate_boundaries(lo: float, point: float, expect: str) -> None:
    assert gate(lo, point, DEFAULT_MARGIN) == expect
