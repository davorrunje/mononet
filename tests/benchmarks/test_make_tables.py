from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarks._common.make_tables import render_verdict

if TYPE_CHECKING:
    from pathlib import Path


def _rec(
    dataset: str, flavor: str, values: list[float], depth: int = 2
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "flavor": flavor,
        "test_metric": "roc_auc",
        "test_values": values,
        "n_collapse": 0,
        "n_seeds": len(values),
        "best_params": {"depth": depth},
        "cv_best": sum(values) / len(values),
    }


def test_render_verdict_reports_alternate_win(tmp_path: Path) -> None:
    d = {
        "split-plain": _rec("heart", "split-plain", [0.70, 0.71, 0.69]),
        "mixed-plain": _rec("heart", "mixed-plain", [0.72, 0.73, 0.71]),
        "alternate-plain": _rec("heart", "alternate-plain", [0.80, 0.81, 0.79]),
    }
    line = render_verdict("heart", d, lower=False)
    assert "alternate" in line
    assert "heart" in line
    # alternate clearly beats best-of-others -> verdict says beats/helps
    assert ("beats" in line.lower()) or ("helps" in line.lower())
