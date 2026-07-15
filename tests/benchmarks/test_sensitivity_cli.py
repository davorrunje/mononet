import pytest

pytest.importorskip("optuna")
from benchmarks.sensitivity import saturation_table


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
