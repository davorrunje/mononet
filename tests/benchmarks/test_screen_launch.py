from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.screen_launch import merge_screens

if TYPE_CHECKING:
    from pathlib import Path


def test_merge_screens_concatenates_and_sorts_by_name(tmp_path: Path) -> None:
    """merge_screens reads per-dataset screen records and sorts by name."""
    p1 = tmp_path / "taiwan.json"
    p1.write_text(
        json.dumps(
            {
                "name": "taiwan",
                "n_full": 24000,
                "deep_iqm": 0.81,
                "shallow_iqm": 0.80,
                "delta": 0.01,
                "delta_lo": -0.01,
                "delta_hi": 0.03,
                "margin": 0.01,
                "verdict": "standard",
            }
        )
    )
    p2 = tmp_path / "adult.json"
    p2.write_text(
        json.dumps(
            {
                "name": "adult",
                "n_full": 32000,
                "deep_iqm": 0.86,
                "shallow_iqm": 0.83,
                "delta": 0.03,
                "delta_lo": 0.01,
                "delta_hi": 0.05,
                "margin": 0.01,
                "verdict": "ladder",
            }
        )
    )
    recs = merge_screens([p1, p2])
    assert [r["name"] for r in recs] == ["adult", "taiwan"]
    assert recs[0]["verdict"] == "ladder"
    assert recs[1]["verdict"] == "standard"
