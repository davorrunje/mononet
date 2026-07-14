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
    convex_fraction: float | None = None,
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
    if convex_fraction is not None:
        rec["best_params"]["convex_fraction"] = convex_fraction
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
    mixed_line = next(
        line for line in out.splitlines() if "| mixed " in line and "5.00" in line
    )
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


def test_render_detailed_shows_convex_fraction_for_mixed_only(tmp_path: Path) -> None:
    _write(
        tmp_path,
        [
            _rec("heart", "split-plain", [0.70, 0.71, 0.69], n_train=200),
            _rec(
                "heart",
                "mixed-plain",
                [0.72, 0.73, 0.71],
                n_train=200,
                convex_fraction=0.35,
            ),
            _rec("heart", "alternate-plain", [0.68, 0.69, 0.67], n_train=200),
        ],
    )
    out = render_detailed(tmp_path)
    mixed_line = next(line for line in out.splitlines() if "| mixed " in line)
    split_line = next(line for line in out.splitlines() if "| split " in line)
    alt_line = next(line for line in out.splitlines() if "| alternate " in line)
    assert "0.35" in mixed_line
    assert "·" in split_line
    assert "·" in alt_line


def test_render_detailed_mixed_fixed_flavor_renders_as_mixed_fix(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        [
            _rec("heart", "split-plain", [0.70, 0.71, 0.69], n_train=200),
            _rec(
                "heart",
                "mixed-plain",
                [0.72, 0.73, 0.71],
                n_train=200,
                convex_fraction=0.35,
            ),
            _rec(
                "heart",
                "mixed-fixed-plain",
                [0.95, 0.96, 0.94],
                n_train=200,
                convex_fraction=0.5,
            ),
            _rec("heart", "alternate-plain", [0.68, 0.69, 0.67], n_train=200),
        ],
    )
    out = render_detailed(tmp_path)
    fixed_line = next(line for line in out.splitlines() if "| mixed-fix " in line)
    assert "🥇" in fixed_line
    assert "0.50" in fixed_line


def test_render_detailed_mixed_fixed_cvxf_defaults_to_050_when_unsearched(
    tmp_path: Path,
) -> None:
    # mixed-fixed-plain pins convex_fraction=0.5 by construction and does not
    # search it, so best_params has no "convex_fraction" key. The cvxf cell
    # must still read 0.50 (not "·"), so it's visibly distinct from
    # split/alternate (which genuinely have no convex_fraction) and contrasts
    # with searched mixed's own value.
    _write(
        tmp_path,
        [
            _rec("heart", "split-plain", [0.70, 0.71, 0.69], n_train=200),
            _rec(
                "heart",
                "mixed-plain",
                [0.72, 0.73, 0.71],
                n_train=200,
                convex_fraction=0.35,
            ),
            _rec(
                "heart",
                "mixed-fixed-plain",
                [0.95, 0.96, 0.94],
                n_train=200,
            ),
            _rec("heart", "alternate-plain", [0.68, 0.69, 0.67], n_train=200),
        ],
    )
    out = render_detailed(tmp_path)
    fixed_line = next(line for line in out.splitlines() if "| mixed-fix " in line)
    mixed_line = next(
        line
        for line in out.splitlines()
        if "| mixed " in line and "mixed-fix" not in line
    )
    split_line = next(line for line in out.splitlines() if "| split " in line)
    alt_line = next(line for line in out.splitlines() if "| alternate " in line)
    assert "0.50" in fixed_line
    assert "0.35" in mixed_line
    assert "·" in split_line
    assert "·" in alt_line


def test_render_detailed_mixed_fixed_absent_renders_no_row(tmp_path: Path) -> None:
    # mixed-fixed-plain is an opt-in extra flavor; when no record exists for it,
    # render_detailed must NOT synthesize a pending "_running_" row for it (unlike
    # split/mixed/alternate, which always render even when missing).
    _write(
        tmp_path,
        [
            _rec("compas", "split-plain", [0.60, 0.61, 0.59], n_train=1000),
            _rec("compas", "mixed-plain", [0.62, 0.63, 0.61], n_train=1000),
            _rec("compas", "alternate-plain", [0.64, 0.65, 0.63], n_train=1000),
        ],
    )
    out = render_detailed(tmp_path)
    assert not any("| mixed-fix" in line for line in out.splitlines())


def test_render_detailed_sorts_datasets_by_n_train_ascending(tmp_path: Path) -> None:
    # "auto" is first in the legacy _ORDER but has the larger n_train here;
    # "blog" is last in _ORDER but has the smaller n_train, so it must render
    # first now that rows are sorted by size.
    _write(
        tmp_path,
        [
            _rec("auto", "split-plain", [10.0, 11.0, 9.0], n_train=50000),
            _rec("blog", "split-plain", [1.0, 1.1, 0.9], n_train=100),
        ],
    )
    out = render_detailed(tmp_path)
    lines = out.splitlines()
    blog_idx = next(i for i, line in enumerate(lines) if line.startswith("| blog ("))
    auto_idx = next(i for i, line in enumerate(lines) if line.startswith("| auto ("))
    assert blog_idx < auto_idx


def test_render_detailed_falls_back_to_order_without_n_train(tmp_path: Path) -> None:
    # Legacy records (no n_train anywhere) must preserve the fixed _ORDER:
    # "auto" before "blog".
    _write(
        tmp_path,
        [
            _rec("blog", "split-plain", [1.0, 1.1, 0.9]),
            _rec("auto", "split-plain", [10.0, 11.0, 9.0]),
        ],
    )
    out = render_detailed(tmp_path)
    lines = out.splitlines()
    auto_idx = next(i for i, line in enumerate(lines) if line.startswith("| auto ("))
    blog_idx = next(i for i, line in enumerate(lines) if line.startswith("| blog ("))
    assert auto_idx < blog_idx
