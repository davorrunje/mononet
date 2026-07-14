"""Pure-function tests for the flavor-ablation report (synthetic records)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks._common.flavor_ablation_report import (
    divergence_by_depth,
    flavor_label,
    flavor_table,
    init_study_table,
    render_divergence_plot,
    verdicts,
)

if TYPE_CHECKING:
    from pathlib import Path


def _rec(
    dataset: str,
    mode: str,
    alt_init: str | None,
    activation: str,
    depth: int,
    *,
    iqm: float,
    div: float,
    collapsed: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "backend": "torch",
        "topology": "plain",
        "mode": mode,
        "alt_init": alt_init,
        "activation": activation,
        "depth": depth,
        "lr": 1e-3,
        "primary": "mse" if dataset.startswith("synth") else "roc_auc",
        "collapsed": collapsed,
        "metric_iqm": iqm,
        "metric_iqr": 0.01,
        "epochs_median": 40.0,
        "divergence_rate": div,
        "n_seeds": 5,
    }


def _records() -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []
    for depth in (4, 8, 16):
        # alternate-composition trains at every depth; mixed/split diverge deep
        deep_div = 1.0 if depth == 16 else 0.0
        recs.append(
            _rec(
                "synth_lattice_cmid",
                "mixed",
                None,
                "relu",
                depth,
                iqm=0.2,
                div=deep_div,
            )
        )
        recs.append(
            _rec(
                "synth_lattice_cmid",
                "split",
                None,
                "relu",
                depth,
                iqm=0.25,
                div=deep_div,
            )
        )
        recs.append(
            _rec(
                "synth_lattice_cmid",
                "alternate",
                "composition",
                "relu",
                depth,
                iqm=0.15,
                div=0.0,
            )
        )
        # legacy collapses for relu at every depth; composition does not
        recs.append(
            _rec(
                "synth_lattice_cmid",
                "alternate",
                "legacy",
                "relu",
                depth,
                iqm=1.0,
                div=1.0,
                collapsed=True,
            )
        )
    return recs


def test_flavor_label() -> None:
    assert flavor_label("mixed", None) == "mixed"
    assert flavor_label("split", None) == "split"
    assert flavor_label("alternate", "composition") == "alternate-composition"
    assert flavor_label("alternate", "legacy") == "alternate-legacy"


def test_flavor_table_has_flavors_and_depths() -> None:
    md = flavor_table(_records(), "synth_lattice_cmid")
    assert "alternate-composition" in md
    assert "mixed" in md
    # depth columns present
    for d in ("4", "8", "16"):
        assert d in md


def test_init_study_table_contrasts_composition_and_legacy() -> None:
    md = init_study_table(_records(), "synth_lattice_cmid")
    assert "composition" in md
    assert "legacy" in md


def test_divergence_by_depth_groups_by_flavor() -> None:
    dbd = divergence_by_depth(_records())
    assert dbd["alternate-composition"][16] == 0.0
    assert dbd["mixed"][16] == 1.0


def test_verdicts_reads_h_plain() -> None:
    v = verdicts(_records())
    # H-plain: alternate-composition diverges less than mixed at depth 16
    assert "H-plain" in v
    assert "supported" in v["H-plain"].lower()


def test_render_divergence_plot_writes_files(tmp_path: Path) -> None:
    render_divergence_plot(_records(), tmp_path / "div")
    assert (tmp_path / "div.png").exists()
    assert (tmp_path / "div.pdf").exists()
