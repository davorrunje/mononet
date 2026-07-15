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
_DETAIL_FLAVS = ["split-plain", "mixed-plain", "mixed-fixed-plain", "alternate-plain"]
# "mixed-fixed-plain" is an opt-in extra flavor (mixed with convex_fraction
# pinned at 0.5); unlike the other detailed flavors it renders no row at all
# when absent, rather than a pending "_running_" placeholder (see
# render_detailed), so older/partial result sets don't sprout a permanent
# pending row for a flavor that was never run.
_DETAIL_OPTIONAL_FLAVS = {"mixed-fixed-plain"}
_DETAIL_LABELS = {"mixed-fixed-plain": "mixed-fix"}


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


def _load(root: Path | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    root = root or (Path(__file__).resolve().parents[1] / "results" / "phase2")
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


def render_verdict(ds: str, d: dict[str, dict[str, Any]], lower: bool) -> str:
    """One-line bootstrap verdict: alternate vs best-of-others plain.

    "Others" is the best non-alternate flavor present: ``split``, ``mixed``,
    and ``mixed-fixed`` (the latter only when its record exists).

    :param ds: Dataset name.
    :param d: Flavor -> record map for this dataset.
    :param lower: Whether lower metric is better.
    :returns: A Markdown table row ``| ds | Δ | 95% CI | verdict |``.
    """
    from benchmarks._common.results import bootstrap_delta

    alt = d.get("alternate-plain")
    raw_others = [
        d.get("split-plain"),
        d.get("mixed-plain"),
        d.get("mixed-fixed-plain"),
    ]
    others: list[dict[str, Any]] = [o for o in raw_others if o]
    if alt is None or not others:
        return f"| {ds} | — | — | *pending* |"
    best_other = (min if lower else max)(others, key=lambda o: _stats(o, ds)[3])
    av = np.asarray(alt["test_values"], np.float64)
    bv = np.asarray(best_other["test_values"], np.float64)
    if ds == "blog":  # values stored as MSE; table reports RMSE
        av, bv = np.sqrt(av), np.sqrt(bv)
    point, lo, hi = bootstrap_delta(av, bv, lower_is_better=lower)
    if lo > 0:
        verdict = "alternate **beats** best-of-others"
    elif hi < 0:
        verdict = "alternate loses"
    else:
        verdict = "matches (CI straddles 0)"
    return (
        f"| {ds} | {point:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {verdict} "
        f"(vs {best_other['flavor']}) |"
    )


def _render_verdict_section(
    rows: dict[str, dict[str, dict[str, Any]]],
) -> list[str]:
    """Build the Markdown "Verdict" section: alternate vs best-of-others.

    :param rows: Dataset -> flavor -> record map, as returned by :func:`_load`.
    :returns: Lines to append to the rendered tables.
    """
    out = ["", "### Verdict — alternate vs best-of-others", ""]
    out.append("| dataset | Δ (alt vs best-other) | 95% CI | verdict |")
    out.append("|---|--:|:--|:--|")
    wins = 0
    for ds in _ORDER:
        d = rows.get(ds, {})
        if not d:
            continue
        lower = _DISP[ds][0] in ("MSE", "RMSE")
        line = render_verdict(ds, d, lower)
        out.append(line)
        if "beats" in line:
            wins += 1
    out.append("")
    out.append(
        f"**alternate beats best-of-others on {wins} of {len(_ORDER)} datasets.**"
    )
    return out


def _detail_winner(d: dict[str, dict[str, Any]], ds: str, lower: bool) -> str | None:
    """Flavor with the best IQM among the present detailed-table flavors."""
    present = [(fl, d[fl]) for fl in _DETAIL_FLAVS if d.get(fl)]
    if not present:
        return None
    scored = [(fl, _stats(r, ds)[3]) for fl, r in present]
    return (min if lower else max)(scored, key=lambda t: t[1])[0]


def _dataset_n_train(d: dict[str, dict[str, Any]]) -> int | None:
    """``n_train`` from the first :data:`_DETAIL_FLAVS` record that has it.

    Checks all three detailed flavors (not just the first present one), since
    older result JSONs may carry ``n_train`` on some flavors but not others.
    """
    for fl in _DETAIL_FLAVS:
        r = d.get(fl)
        if r is not None and r.get("n_train") is not None:
            return int(r["n_train"])
    return None


def _detail_rows_cell(d: dict[str, dict[str, Any]]) -> str:
    """``n_train`` (with thousands separator), or ``—`` if none of the records have it."""
    n_train = _dataset_n_train(d)
    return f"{n_train:,}" if n_train is not None else "—"


def _detail_hp_cells(bp: dict[str, Any], fl: str) -> list[str]:
    """``act/layers/width/lr/wdec/drop/lrdec/batch`` cells from ``best_params``.

    ``activation`` is guarded with a fallback: it is absent from result JSONs
    written before the activation search was added (e.g. ``phase2``).
    """
    return [
        str(bp.get("activation", "—")),
        str(_layers(fl, bp["depth"])),
        str(bp["width"]),
        f"{bp['lr']:.4f}",
        f"{bp['weight_decay']:.3f}",
        f"{bp['dropout']:.2f}",
        f"{bp['lr_decay']:.3f}",
        str(bp["batch_size"]),
    ]


def _detail_cvxf_cell(bp: dict[str, Any], fl: str) -> str:
    """``convex_fraction`` from ``best_params``, formatted ``.2f`` or ``·`` if absent.

    Only the ``mixed`` flavor searches ``convex_fraction``; ``split`` and
    ``alternate`` do not, so they render the ``·`` placeholder. ``mixed-fixed``
    pins ``convex_fraction=0.5`` by construction and never searches it (so it
    is absent from ``best_params``), but the cell still renders ``0.50`` to
    make that pinning visible and distinguish it from ``split``/``alternate``.
    """
    if fl == "mixed-fixed-plain":
        return "0.50"
    cvxf = bp.get("convex_fraction")
    return f"{cvxf:.2f}" if cvxf is not None else "·"


def _detail_flavor_label(fl: str) -> str:
    """Display name for a detailed-table flavor (bare, no medal/bold)."""
    return _DETAIL_LABELS.get(fl, fl.removesuffix("-plain"))


def _detail_row(
    ds: str,
    label: str,
    rows_cell: str,
    fl: str,
    r: dict[str, Any] | None,
    is_winner: bool,
) -> str:
    """Render one row of the detailed table for a single ``(dataset, flavor)``."""
    name = _detail_flavor_label(fl)
    if r is None:
        blanks = [""] * 10
        cells = [label, rows_cell, name, "_running_", *blanks, "⏳"]
        return "| " + " | ".join(cells) + " |"
    me, sd, _, iqm = _stats(r, ds)
    iqmc = _num(iqm, ds)
    if is_winner:
        name = f"{name} 🥇"
        iqmc = f"**{iqmc}**"
    cells = [
        label,
        rows_cell,
        name,
        iqmc,
        f"{_num(me, ds)} ± {_num(sd, ds)}",
        *_detail_hp_cells(r["best_params"], fl),
        _detail_cvxf_cell(r["best_params"], fl),
        "✅",
    ]
    return "| " + " | ".join(cells) + " |"


def _dataset_order(rows: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    """Datasets ordered by ascending ``n_train``, falling back to :data:`_ORDER`.

    Datasets with a known ``n_train`` sort first (smallest first); datasets
    with no ``n_train`` in any record (older result JSONs) keep their
    original :data:`_ORDER` position, so legacy data still renders in the
    same order it always has.

    :param rows: Dataset -> flavor -> record map, as returned by :func:`_load`.
    :returns: Dataset names in render order.
    """

    def _key(ds: str) -> tuple[int, int]:
        n_train = _dataset_n_train(rows[ds])
        if n_train is not None:
            return (0, n_train)
        return (1, _ORDER.index(ds) if ds in _ORDER else len(_ORDER))

    return sorted(rows, key=_key)


def render_detailed(root: Path | None = None) -> str:
    """Return a detailed per-flavor Markdown table (data size, HPs, winner medal).

    One row per ``(dataset, flavor)`` for the plain-only flavors ``split``,
    ``mixed``, ``mixed-fixed`` (mixed with ``convex_fraction`` pinned at
    ``0.5``), ``alternate``. Datasets are ordered by ascending ``n_train``
    (smallest first), falling back to the fixed :data:`_ORDER` for datasets
    with no ``n_train`` recorded. The per-dataset best IQM (direction taken
    from :data:`_DISP`) is marked with a 🥇 and bolded; flavors missing from
    the result JSONs (partial runs) render as ``_running_`` / ``⏳`` — except
    ``mixed-fixed``, an opt-in extra flavor whose row is omitted entirely
    when no record exists for it (see :data:`_DETAIL_OPTIONAL_FLAVS`).

    :param root: Directory containing per-flavor result JSONs. Defaults to
        ``benchmarks/results/phase2``.
    :returns: The rendered Markdown table.
    """
    rows = _load(root)
    out = [
        "| dataset | rows | flavor | IQM | mean ± std | act | layers | width "
        "| lr | wdec | drop | lrdec | batch | cvxf | done |",
        "|---|--:|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|",
    ]
    for ds in _dataset_order(rows):
        d = rows.get(ds, {})
        if not d:
            continue
        m, arrow = _DISP[ds]
        lower = m in ("MSE", "RMSE")
        winner = _detail_winner(d, ds, lower)
        rows_cell = _detail_rows_cell(d)
        for i, fl in enumerate(_DETAIL_FLAVS):
            r = d.get(fl)
            if r is None and fl in _DETAIL_OPTIONAL_FLAVS:
                continue
            label = f"{ds} ({m} {arrow})" if i == 0 else ""
            rc = rows_cell if i == 0 else ""
            out.append(_detail_row(ds, label, rc, fl, r, fl == winner))
    return "\n".join(out)


def render(root: Path | None = None) -> str:
    """Return the main + robustness Markdown tables as one string.

    :param root: Directory containing per-flavor result JSONs. Defaults to
        ``benchmarks/results/phase2``.
    :returns: The rendered Markdown tables.
    """
    rows = _load(root)
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
            ("alternate", "plain", d.get("alternate-plain")),
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

    out += _render_verdict_section(rows)

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
    """Print the rendered tables to stdout.

    Pass ``--detailed`` to print :func:`render_detailed` instead of the
    default compact :func:`render` output.
    """
    import sys

    if "--detailed" in sys.argv[1:]:
        print(render_detailed())  # noqa: T201
    else:
        print(render())  # noqa: T201


if __name__ == "__main__":
    main()
