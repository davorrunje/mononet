from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks._common.screen_report import screen_table

if TYPE_CHECKING:
    from pathlib import Path


def _rec(name: str, delta: float, lo: float, hi: float, verdict: str) -> dict[str, Any]:
    return {
        "name": name,
        "n_full": 1000,
        "deep_iqm": 0.7,
        "shallow_iqm": 0.7 - delta,
        "delta": delta,
        "delta_lo": lo,
        "delta_hi": hi,
        "margin": 0.005,
        "verdict": verdict,
    }


def test_screen_table_has_row_per_dataset_and_verdict() -> None:
    rows = [
        _rec("a", 0.01, 0.006, 0.014, "ladder"),
        _rec("b", 0.0, -0.003, 0.003, "standard"),
    ]
    md = screen_table(rows)
    assert "| a |" in md
    assert "| b |" in md
    assert "ladder" in md
    assert "standard" in md


def test_render_screen_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    import pytest

    pytest.importorskip("matplotlib")
    from benchmarks._common.screen_report import render_screen_plot

    rows = [
        _rec("a", 0.01, 0.006, 0.014, "ladder"),
        _rec("b", 0.0, -0.003, 0.003, "standard"),
    ]
    render_screen_plot(rows, tmp_path / "screen.png")
    for suffix in (".png", ".pdf"):
        out = (tmp_path / "screen").with_suffix(suffix)
        assert out.exists()
        assert out.stat().st_size > 0
