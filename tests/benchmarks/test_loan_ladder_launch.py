from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.loan_ladder_launch import merge_partials

if TYPE_CHECKING:
    from pathlib import Path


def test_merge_partials_concatenates_and_sorts(tmp_path: Path) -> None:
    """merge_partials concatenates per-cell record lists, ordered by (n, arm)."""
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps([{"n": 400, "arm": "deep", "test_iqm": 0.60}]))
    p2 = tmp_path / "b.json"
    p2.write_text(
        json.dumps(
            [
                {"n": 100, "arm": "shallow", "test_iqm": 0.50},
                {"n": 100, "arm": "deep", "test_iqm": 0.55},
            ]
        )
    )
    recs = merge_partials([p1, p2])
    assert [(r["n"], r["arm"]) for r in recs] == [
        (100, "deep"),
        (100, "shallow"),
        (400, "deep"),
    ]
