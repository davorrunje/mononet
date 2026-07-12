from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from benchmarks._common.depth_probe_report import delta_by_c, probe_table

if TYPE_CHECKING:
    from pathlib import Path


def _rec(kind: str, c: int, deep: list[float], shallow: list[float]) -> dict[str, Any]:
    import numpy as np

    from benchmarks._common.results import interquartile_mean

    return {
        "kind": kind,
        "c": c,
        "deep_mse_iqm": interquartile_mean(np.asarray(deep)),
        "shallow_mse_iqm": interquartile_mean(np.asarray(shallow)),
        "deep_values": deep,
        "shallow_values": shallow,
    }


def test_delta_by_c_sign_and_band() -> None:
    """Deep clearly better (lower MSE) ⇒ positive delta, band brackets it."""
    rows = delta_by_c([_rec("teacher", 4, [0.10, 0.11, 0.10], [0.20, 0.21, 0.20])])
    r = rows[0]
    assert r["delta"] > 0
    assert r["delta_lo"] <= r["delta"] <= r["delta_hi"]


def test_delta_by_c_shallow_better_gives_negative_delta() -> None:
    """Shallow clearly better (lower MSE) ⇒ negative delta."""
    rows = delta_by_c([_rec("lattice", 2, [0.30, 0.31, 0.30], [0.10, 0.11, 0.10])])
    r = rows[0]
    assert r["delta"] < 0
    assert r["delta_lo"] <= r["delta"] <= r["delta_hi"]


def test_delta_by_c_preserves_kind_and_c() -> None:
    rows = delta_by_c([_rec("additive", 3, [0.2, 0.2], [0.2, 0.2])])
    assert rows[0]["kind"] == "additive"
    assert rows[0]["c"] == 3


def test_probe_table_has_rows() -> None:
    rows = delta_by_c([_rec("additive", 1, [0.2, 0.2], [0.2, 0.2])])
    md = probe_table(rows)
    assert "| additive |" in md


def test_render_probe_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    """render_probe_plot writes both a PNG (docs) and a vector PDF (LaTeX)."""
    pytest.importorskip("matplotlib")
    from benchmarks._common.depth_probe_report import render_probe_plot

    records = [
        _rec("additive", 1, [0.20, 0.21, 0.20], [0.20, 0.20, 0.21]),
        _rec("additive", 2, [0.15, 0.16, 0.15], [0.25, 0.26, 0.25]),
        _rec("teacher", 1, [0.30, 0.31, 0.30], [0.10, 0.11, 0.10]),
    ]
    rows = delta_by_c(records)
    render_probe_plot(rows, tmp_path / "depth-probe.png")
    for suffix in (".png", ".pdf"):
        out = (tmp_path / "depth-probe").with_suffix(suffix)
        assert out.exists()
        assert out.stat().st_size > 0
