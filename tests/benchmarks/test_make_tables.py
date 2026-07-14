from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from benchmarks._common.make_tables import render_detailed, render_verdict

if TYPE_CHECKING:
    from pathlib import Path


def _rec(
    dataset: str,
    flavor: str,
    values: list[float],
    depth: int = 2,
    *,
    n_train: int | None = None,
    activation: str = "relu",
    width: int = 32,
    lr: float = 0.01,
    weight_decay: float = 0.0001,
    dropout: float = 0.1,
    lr_decay: float = 0.9,
    batch_size: int = 64,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "dataset": dataset,
        "flavor": flavor,
        "test_metric": "roc_auc",
        "test_values": values,
        "n_collapse": 0,
        "n_seeds": len(values),
        "best_params": {
            "activation": activation,
            "depth": depth,
            "width": width,
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "lr_decay": lr_decay,
            "batch_size": batch_size,
        },
        "cv_best": sum(values) / len(values),
    }
    if n_train is not None:
        rec["n_train"] = n_train
    return rec


def _write(root: Path, recs: list[dict[str, Any]]) -> None:
    for r in recs:
        path = root / f"{r['dataset']}-{r['flavor']}.json"
        path.write_text(json.dumps(r))


def test_render_verdict_reports_alternate_win() -> None:
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


def test_render_detailed_marks_winner_and_bolds_iqm(tmp_path: Path) -> None:
    # auto: MSE, lower is better -> mixed-plain (lowest values) should win.
    _write(
        tmp_path,
        [
            _rec("auto", "split-plain", [10.0, 11.0, 9.0], n_train=300),
            _rec("auto", "mixed-plain", [5.0, 5.5, 4.5], n_train=300),
            _rec("auto", "alternate-plain", [8.0, 8.5, 7.5], n_train=300),
        ],
    )
    # heart: accuracy, higher is better -> alternate-plain (highest values) wins.
    _write(
        tmp_path,
        [
            _rec("heart", "split-plain", [0.70, 0.71, 0.69], n_train=200),
            _rec("heart", "mixed-plain", [0.72, 0.73, 0.71], n_train=200),
            _rec("heart", "alternate-plain", [0.90, 0.91, 0.89], n_train=200),
        ],
    )
    out = render_detailed(tmp_path)

    # mixed-plain wins on auto (lower-is-better).
    mixed_line = next(line for line in out.splitlines() if "| mixed " in line)
    assert "🥇" in mixed_line
    assert "**" in mixed_line
    split_auto_line = next(
        line
        for line in out.splitlines()
        if line.startswith("| auto") and "| split " in line
    )
    assert "🥇" not in split_auto_line

    # alternate-plain wins on heart (higher-is-better).
    alt_line = next(
        line for line in out.splitlines() if "| alternate 🥇" in line and "0.9" in line
    )
    assert "**" in alt_line


def test_render_detailed_marks_missing_flavor_as_running(tmp_path: Path) -> None:
    # Only split-plain and mixed-plain present; alternate-plain missing (partial run).
    _write(
        tmp_path,
        [
            _rec("compas", "split-plain", [0.60, 0.61, 0.59], n_train=1000),
            _rec("compas", "mixed-plain", [0.62, 0.63, 0.61], n_train=1000),
        ],
    )
    out = render_detailed(tmp_path)
    running_line = next(line for line in out.splitlines() if "| alternate " in line)
    assert "_running_" in running_line
    assert "⏳" in running_line


def test_render_detailed_includes_rows_and_hyperparameters(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [
            _rec(
                "loan",
                "split-plain",
                [0.80, 0.81, 0.79],
                depth=3,
                n_train=12345,
                width=64,
                lr=0.005,
                batch_size=128,
            ),
            _rec("loan", "mixed-plain", [0.82, 0.83, 0.81], n_train=12345),
            _rec("loan", "alternate-plain", [0.78, 0.79, 0.77], n_train=12345),
        ],
    )
    out = render_detailed(tmp_path)
    assert "12,345" in out
    split_line = next(line for line in out.splitlines() if "| split " in line)
    assert "64" in split_line
    assert "128" in split_line
    assert "0.0050" in split_line
