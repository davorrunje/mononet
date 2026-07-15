import pytest

pytest.importorskip("optuna")
from pathlib import Path

from benchmarks.sensitivity import render_from_curves, saturation_table


def test_saturation_table_is_markdown_with_status_column() -> None:
    rows = [
        {
            "dataset": "heart",
            "flavor": "split-plain",
            "trials": 200,
            "t_star": 40,
            "saturated": True,
            "n_reeval": 5,
        },
        {
            "dataset": "loan",
            "flavor": "mixed-plain",
            "trials": 50,
            "t_star": 49,
            "saturated": False,
            "n_reeval": 3,
        },
    ]
    md = saturation_table(rows)
    assert md.startswith("| dataset | flavor | trials | t*(0.99) |")
    assert "| heart | split-plain | 200 | 40 | ✅ | 5 |" in md
    assert "| loan | mixed-plain | 50 | 49 | ⚠️ | 3 |" in md


def test_render_from_curves_writes_figure_and_orders_rows(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    # auto before compas in the output despite input order (sorted by _ORDER).
    curves = [
        {
            "dataset": "compas",
            "flavors": {"split-plain": {"obj": [0.7, 0.72], "test": [0.69, 0.70]}},
            "rows": [
                {
                    "dataset": "compas",
                    "flavor": "split-plain",
                    "trials": 50,
                    "t_star": 30,
                    "saturated": True,
                    "n_reeval": 4,
                }
            ],
        },
        {
            "dataset": "auto",
            "flavors": {"mixed-plain": {"obj": [10.5, 10.1], "test": None}},
            "rows": [
                {
                    "dataset": "auto",
                    "flavor": "mixed-plain",
                    "trials": 200,
                    "t_star": 11,
                    "saturated": True,
                    "n_reeval": 0,
                }
            ],
        },
    ]
    out = tmp_path / "sens"
    md = render_from_curves(curves, out)
    assert out.with_suffix(".png").stat().st_size > 0
    assert out.with_suffix(".pdf").stat().st_size > 0
    # _ORDER puts auto (row) before compas.
    assert md.index("| auto |") < md.index("| compas |")
