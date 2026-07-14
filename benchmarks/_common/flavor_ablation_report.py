"""Render the flavor-ablation study pages from committed results JSON.

Reads the per-dataset ``<dataset>.json`` records written by
:mod:`benchmarks.flavor_ablation` and produces three markdown studies —
**flavor** (mixed vs alternate-composition vs split), **initialization**
(alternate composition vs legacy), and **residual** (scope note) — plus a
divergence-vs-depth figure. The hypotheses (H-plain / H-init / H-residual) are
read off the numbers by :func:`verdicts`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

_ACT_ORDER = ("relu", "elu", "softplus", "selu")
_DEPTHS = (4, 8, 16)
_FLAVOR_ORDER = ("mixed", "split", "alternate-composition", "alternate-legacy")
_DIVERGED_MARK = "✗"  # ✗ — appended when divergence_rate > 0.5


def flavor_label(mode: str, alt_init: str | None) -> str:
    """Human-readable flavor label from ``(mode, alt_init)``.

    :param mode: Construction mode.
    :param alt_init: Alternate init arm, or ``None``.
    :returns: e.g. ``"mixed"`` or ``"alternate-composition"``.
    """
    return f"{mode}-{alt_init}" if alt_init else mode


def load_records(results_dir: Path, *, lr_sweep: bool = False) -> list[dict[str, Any]]:
    """Load all cell records from ``results_dir``.

    :param results_dir: Directory of ``<dataset>[-lrsweep].json`` files.
    :param lr_sweep: If True, read the ``*-lrsweep.json`` files; else the main
        ``<dataset>.json`` files.
    :returns: The concatenated record list.
    """
    recs: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        is_sweep = path.stem.endswith("-lrsweep")
        if is_sweep != lr_sweep:
            continue
        recs.extend(json.loads(path.read_text()))
    return recs


def _label(rec: dict[str, Any]) -> str:
    return flavor_label(rec["mode"], rec["alt_init"])


def _cell(rec: dict[str, Any]) -> str:
    mark = _DIVERGED_MARK if rec["divergence_rate"] > 0.5 else ""
    iqm = rec["metric_iqm"]
    return f"{iqm:.3f}{mark}" if iqm == iqm else "nan"


def flavor_table(records: list[dict[str, Any]], dataset: str) -> str:
    """Markdown table: primary metric by (flavor x activation) rows, depth cols.

    A ``✗`` marks cells whose divergence-rate exceeds 0.5.

    :param records: All cell records.
    :param dataset: The dataset to tabulate.
    :returns: A GitHub-flavored markdown table.
    """
    rows = [r for r in records if r["dataset"] == dataset]
    primary = rows[0]["primary"] if rows else "metric"
    by_key = {(_label(r), r["activation"], r["depth"]): r for r in rows}
    header = "| flavor | activation | " + " | ".join(f"d{d}" for d in _DEPTHS) + " |"
    sep = "|---|---|" + "---|" * len(_DEPTHS)
    lines = [
        f"Primary metric: **{primary}** (`✗` = divergence-rate > 0.5)",
        "",
        header,
        sep,
    ]
    for flavor in _FLAVOR_ORDER:
        for act in _ACT_ORDER:
            cells = []
            present = False
            for d in _DEPTHS:
                rec = by_key.get((flavor, act, d))
                if rec is None:
                    cells.append("-")
                else:
                    present = True
                    cells.append(_cell(rec))
            if present:
                lines.append(f"| {flavor} | {act} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def init_study_table(records: list[dict[str, Any]], dataset: str) -> str:
    """Markdown table contrasting alternate **composition** vs **legacy** init.

    Per (activation, depth): the divergence-rate and init-collapse flag for each
    arm — the collapse the composition-aware init is meant to prevent.

    :param records: All cell records.
    :param dataset: The dataset to tabulate.
    :returns: A GitHub-flavored markdown table.
    """
    rows = [r for r in records if r["dataset"] == dataset and r["mode"] == "alternate"]
    by_key = {(r["alt_init"], r["activation"], r["depth"]): r for r in rows}
    header = (
        "| activation | depth | composition div | legacy div "
        "| composition collapsed | legacy collapsed |"
    )
    sep = "|---|---|---|---|---|---|"
    lines = [header, sep]
    for act in _ACT_ORDER:
        for d in _DEPTHS:
            comp = by_key.get(("composition", act, d))
            leg = by_key.get(("legacy", act, d))
            if comp is None and leg is None:
                continue
            cd = f"{comp['divergence_rate']:.2f}" if comp else "-"
            ld = f"{leg['divergence_rate']:.2f}" if leg else "-"
            cc = str(comp["collapsed"]) if comp else "-"
            lc = str(leg["collapsed"]) if leg else "-"
            lines.append(f"| {act} | {d} | {cd} | {ld} | {cc} | {lc} |")
    return "\n".join(lines)


def divergence_by_depth(
    records: list[dict[str, Any]],
) -> dict[str, dict[int, float]]:
    """Mean divergence-rate per flavor per depth (averaged over activations).

    :param records: All cell records (a single dataset, or pooled).
    :returns: ``{flavor: {depth: mean_divergence_rate}}``.
    """
    acc: dict[str, dict[int, list[float]]] = {}
    for r in records:
        acc.setdefault(_label(r), {}).setdefault(r["depth"], []).append(
            r["divergence_rate"]
        )
    return {
        flavor: {d: sum(v) / len(v) for d, v in depths.items()}
        for flavor, depths in acc.items()
    }


def verdicts(records: list[dict[str, Any]]) -> dict[str, str]:
    """Read the H-plain / H-init / H-residual verdicts off the records.

    :param records: All main (non-lr-sweep) cell records.
    :returns: ``{hypothesis: verdict-sentence}``.
    """
    dbd = divergence_by_depth(records)
    out: dict[str, str] = {}

    # H-plain: at depth 16, alternate-composition should diverge markedly less
    # than mixed/split.
    ac16 = dbd.get("alternate-composition", {}).get(16)
    baseline16: list[float] = []
    for f in ("mixed", "split"):
        v = dbd.get(f, {}).get(16)
        if v is not None:
            baseline16.append(v)
    if ac16 is not None and baseline16:
        worst = max(baseline16)
        if ac16 + 0.25 < worst:
            out["H-plain"] = (
                f"supported: at depth 16 alternate-composition divergence-rate "
                f"{ac16:.2f} vs mixed/split up to {worst:.2f}."
            )
        else:
            out["H-plain"] = (
                f"not supported here: alternate-composition {ac16:.2f} vs "
                f"mixed/split up to {worst:.2f} at depth 16."
            )

    # H-init: composition should diverge/collapse less than legacy.
    comp = [r for r in records if r["alt_init"] == "composition"]
    leg = [r for r in records if r["alt_init"] == "legacy"]
    if comp and leg:
        comp_div = sum(r["divergence_rate"] for r in comp) / len(comp)
        leg_div = sum(r["divergence_rate"] for r in leg) / len(leg)
        verdict = "supported" if comp_div + 0.1 < leg_div else "not supported here"
        out["H-init"] = (
            f"{verdict}: mean divergence-rate composition {comp_div:.2f} vs "
            f"legacy {leg_div:.2f}."
        )

    out["H-residual"] = (
        "not evaluated: the focused run is plain-topology only "
        "(residual + alternate is a documented expansion)."
    )
    return out


def render_divergence_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    """Render divergence-rate vs depth, one line per flavor (PNG + PDF).

    :param records: All main cell records (pooled over datasets).
    :param out_path: Base path; ``.png`` and ``.pdf`` are written alongside.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dbd = divergence_by_depth(records)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for flavor in _FLAVOR_ORDER:
        depths = dbd.get(flavor)
        if not depths:
            continue
        xs = sorted(depths)
        ax.plot(xs, [depths[d] for d in xs], marker="o", label=flavor)
    ax.set_xlabel("plain-stack depth")
    ax.set_ylabel("divergence-rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(list(_DEPTHS))
    ax.legend(fontsize="small")
    ax.set_title("Divergence-rate vs depth (pooled)")
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_pages(records: list[dict[str, Any]], docs_dir: Path) -> list[Path]:
    """Write the three study markdown pages + the divergence figure.

    :param records: All main cell records.
    :param docs_dir: ``docs/benchmarks`` directory.
    :returns: The written page paths.
    """
    datasets = sorted({r["dataset"] for r in records})
    v = verdicts(records)
    render_divergence_plot(records, docs_dir / "flavor-ablation-divergence")

    flavor_md = [
        "# Flavor study: mixed vs alternate vs split (plain depth)",
        "",
        "Fixed-architecture ablation (width 32, LR 1e-3, 5 seeds, early stopping);",
        "the construction flavor is the only moving part. See the",
        "[spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-07-14-flavor-ablation-benchmark-design.md).",
        "",
        f"**H-plain** — {v.get('H-plain', 'n/a')}",
        "",
        "![divergence vs depth](flavor-ablation-divergence.png)",
        "",
    ]
    for ds in datasets:
        flavor_md += [f"## {ds}", "", flavor_table(records, ds), ""]

    init_md = [
        "# Initialization study: composition-aware vs legacy (alternate)",
        "",
        f"**H-init** — {v.get('H-init', 'n/a')}",
        "",
        '`composition` = the real construction (`mode="alternate"` + `prev=`);',
        '`legacy` = pure convex/concave `mode="mixed"` layers alternating 1/0',
        "(the collapse baseline). `collapsed` is the init-time dead-output check.",
        "",
    ]
    for ds in datasets:
        init_md += [f"## {ds}", "", init_study_table(records, ds), ""]

    residual_md = [
        "# Residual study",
        "",
        f"**H-residual** — {v.get('H-residual', 'n/a')}",
        "",
        "The focused run is plain-topology only: in `MonoResidual` the",
        "near-identity start tames all three flavors, so alternation is expected",
        "to be a wash. Confirming this needs the residual-alternate arm, a",
        "documented expansion not run here.",
        "",
    ]

    pages = {
        "flavor-study.md": "\n".join(flavor_md),
        "initialization-study.md": "\n".join(init_md),
        "residual-study.md": "\n".join(residual_md),
    }
    written: list[Path] = []
    for name, text in pages.items():
        path = docs_dir / name
        path.write_text(text + "\n")
        written.append(path)
    return written


def main() -> None:
    """CLI: render the study pages from a results directory."""
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/results/flavor-ablation"),
    )
    ap.add_argument("--docs-dir", type=Path, default=Path("docs/benchmarks"))
    args = ap.parse_args()
    records = load_records(args.results_dir)
    if not records:
        raise SystemExit(f"no records in {args.results_dir}")
    pages = _write_pages(records, args.docs_dir)
    for p in pages:
        print(f"wrote {p}")  # noqa: T201


if __name__ == "__main__":
    main()
