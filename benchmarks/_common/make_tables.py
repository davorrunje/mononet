"""Render the Stage-2 deep-residual-accuracy result tables as Markdown.

Reads ``benchmarks/results/phase2/*.json`` and emits the two paper-facing
tables (main collapsed plain/residual, and the full-6-flavor robustness table)
plus a one-line best-per-dataset summary. Reproduce with::

    uv run --group bench python -m benchmarks._common.make_tables
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Display metric per dataset (blog reported as RMSE = sqrt(MSE)).
_DISP = {
    "auto": ("MSE", "↓"),
    "heart": ("acc", "↑"),
    "compas": ("acc", "↑"),
    "loan": ("acc", "↑"),
    "blog": ("RMSE", "↓"),
}
_ORDER = ["auto", "heart", "compas", "loan", "blog"]
_FLAVS = [
    "split-plain",
    "split-residual",
    "split-deep",
    "mixed-plain",
    "mixed-residual",
    "mixed-deep",
]


def _layers(flavor: str, depth: int) -> int:
    """Effective monotone layers: plain = depth+1; residual = 1 proj + 2*blocks + head."""
    residual = ("residual" in flavor) or ("deep" in flavor)
    return (2 * depth + 2) if residual else (depth + 1)


def _stats(rec: dict[str, Any], ds: str) -> tuple[float, float, float, float]:
    """(mean, std, median, iqm) over per-seed values; blog mapped to RMSE first."""
    v = np.asarray(rec["test_values"], dtype=np.float64)
    if ds == "blog":
        v = np.sqrt(v)
    s = np.sort(v)
    n = s.size
    k = n // 4
    iqm = float(s[k : n - k].mean()) if n - 2 * k > 0 else float(v.mean())
    return float(v.mean()), float(v.std()), float(np.median(v)), iqm


def _num(x: float, ds: str) -> str:
    return f"{x:.2f}" if ds == "auto" else f"{x:.3f}"


def _load() -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    root = Path(__file__).resolve().parents[1] / "results" / "phase2"
    for f in sorted(root.glob("*.json")):
        r = json.loads(f.read_text())
        out.setdefault(r["dataset"], {})[r["flavor"]] = r
    return out


def _pick_residual(
    d: dict[str, dict[str, Any]], mode: str, lower: bool
) -> dict[str, Any] | None:
    """Collapsed residual = better of {residual, deep} by the stability-aware CV score."""
    cands = [c for c in (d.get(f"{mode}-residual"), d.get(f"{mode}-deep")) if c]
    if not cands:
        return None
    best: dict[str, Any] = (min if lower else max)(cands, key=lambda c: c["cv_best"])
    return best


def render() -> str:
    """Return the main + robustness Markdown tables as one string."""
    rows = _load()
    out: list[str] = ["### Main results (collapsed plain/residual)", ""]
    out.append("| dataset | mode | variant | layers | IQM | mean ± std | ⚠ |")
    out.append("|---|---|---|--:|--:|--:|:-:|")
    for ds in _ORDER:
        d = rows.get(ds, {})
        if not d:
            continue
        m, arrow = _DISP[ds]
        lower = m in ("MSE", "RMSE")
        entries = [
            ("split", "plain", d.get("split-plain")),
            ("split", "residual", _pick_residual(d, "split", lower)),
            ("mixed", "plain", d.get("mixed-plain")),
            ("mixed", "residual", _pick_residual(d, "mixed", lower)),
        ]
        scored = [(e, _stats(e[2], ds)[3], e[2]["n_collapse"]) for e in entries if e[2]]
        best = (
            sorted(scored, key=lambda t: ((t[1] if lower else -t[1]), t[2]))[0][0]
            if scored
            else None
        )
        for i, (mode, var, r) in enumerate(entries):
            label = f"{ds} ({m} {arrow})" if i == 0 else ""
            if not r:
                out.append(f"| {label} | {mode} | {var} | — | *pending* | | |")
                continue
            me, sd, _, iqm = _stats(r, ds)
            dep = r["best_params"]["depth"]
            iqmc = _num(iqm, ds)
            if best is not None and (mode, var) == (best[0], best[1]):
                iqmc = f"**{iqmc}**"
            warn = f"{r['n_collapse']}/{r['n_seeds']}" if r["n_collapse"] else "·"
            out.append(
                f"| {label} | {mode} | {var} | {_layers(r['flavor'], dep)} | "
                f"{iqmc} | {_num(me, ds)} ± {_num(sd, ds)} | {warn} |"
            )

    out += ["", "### Robustness — all six flavors", ""]
    out.append("| dataset | flavor | layers (blocks) | mean ± std | median | IQM | ⚠ |")
    out.append("|---|---|--:|--:|--:|--:|:-:|")
    for ds in _ORDER:
        d = rows.get(ds, {})
        for fl in _FLAVS:
            r = d.get(fl)
            if not r:
                continue
            me, sd, med, iqm = _stats(r, ds)
            dep = r["best_params"]["depth"]
            warn = f"{r['n_collapse']}/{r['n_seeds']}" if r["n_collapse"] else "·"
            out.append(
                f"| {ds} | {fl} | {_layers(fl, dep)} (d{dep}) | "
                f"{_num(me, ds)} ± {_num(sd, ds)} | {_num(med, ds)} | "
                f"{_num(iqm, ds)} | {warn} |"
            )
    return "\n".join(out)


def main() -> None:
    """Print the rendered tables to stdout."""
    print(render())  # noqa: T201


if __name__ == "__main__":
    main()
