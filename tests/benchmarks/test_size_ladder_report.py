from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from benchmarks._common.size_ladder_report import delta_by_n

if TYPE_CHECKING:
    from pathlib import Path


def _rec(
    n: int, arm: str, iqm: float, values: list[float], *, metric: str = "accuracy"
) -> dict[str, Any]:
    return {
        "n": n,
        "arm": arm,
        "test_iqm": iqm,
        "test_values": values,
        "test_metric": metric,
    }


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


def test_delta_by_n_flips_sign_for_lower_is_better_metric() -> None:
    """Mse (lower-is-better): deep with LOWER mse must give a positive Δ.

    Regression datasets like `blog` report `mse` as their primary metric.
    Hardcoding `lower_is_better=False` would flip this sign (deep looking
    worse when it is actually better).
    """
    records = [
        _rec(100, "shallow", 10.0, [10.0, 10.0], metric="mse"),
        _rec(100, "deep", 6.0, [6.0, 6.0], metric="mse"),
    ]
    rows = delta_by_n(records)
    assert rows[0]["delta"] == pytest.approx(4.0, abs=1e-9)  # deep is better


def test_delta_by_n_keeps_higher_is_better_for_roc_auc_and_accuracy() -> None:
    """roc_auc/accuracy (higher-is-better) keep the original (unflipped) sign."""
    for metric in ("roc_auc", "accuracy"):
        records = [
            _rec(100, "shallow", 0.60, [0.60, 0.60], metric=metric),
            _rec(100, "deep", 0.58, [0.58, 0.58], metric=metric),
        ]
        rows = delta_by_n(records)
        assert rows[0]["delta"] == pytest.approx(-0.02, abs=1e-9)


def test_render_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    """render_plot writes both a PNG (docs) and a vector PDF (LaTeX), no raise."""
    pytest.importorskip("matplotlib")
    from benchmarks._common.size_ladder_report import render_plot

    records = [
        _rec(100, "shallow", 0.60, [0.59, 0.61]),
        _rec(100, "deep", 0.58, [0.57, 0.59]),
        _rec(400, "shallow", 0.60, [0.60, 0.60]),
        _rec(400, "deep", 0.66, [0.65, 0.67]),
    ]
    render_plot(records, tmp_path / "loan-size-ladder.png")
    for suffix in (".png", ".pdf"):
        out = (tmp_path / "loan-size-ladder").with_suffix(suffix)
        assert out.exists()
        assert out.stat().st_size > 0
