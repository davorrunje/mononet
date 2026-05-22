"""Placeholder for the cross-backend equivalence harness.

The real harness lands in the follow-up algorithm plan. This file exists
so the per-backend CI matrix has a `tests/equivalence` directory to point
at from day one.
"""

from __future__ import annotations


def test_equivalence_directory_exists() -> None:
    """Smoke test: ensures the equivalence test module is importable."""
    assert True
