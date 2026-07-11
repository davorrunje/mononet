"""Render the max-size screen as a Markdown table + a Δ-per-dataset plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def screen_table(records: list[dict[str, Any]]) -> str:
    """Markdown table: one row per dataset with Δ ± CI and the verdict.

    :param records: List of screen result dicts with keys {name, n_full, deep_iqm, shallow_iqm, delta, delta_lo, delta_hi, margin, verdict}.
    :returns: Markdown-formatted table string.
    """
    head = (
        "| dataset | N | deep IQM | shallow IQM | Δ [95% CI] | verdict |\n"
        "|---|--:|--:|--:|--:|:--|\n"
    )
    rows = [
        f"| {r['name']} | {r['n_full']} | {r['deep_iqm']:.4f} | "
        f"{r['shallow_iqm']:.4f} | {r['delta']:+.4f} "
        f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}] | {r['verdict']} |"
        for r in records
    ]
    return head + "\n".join(rows) + "\n"


def render_screen_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    """Δ per dataset (sorted) with CI bars + 0 and margin reference lines.

    Writes both ``out_path`` (``.png``) and a sibling ``.pdf`` (vector, LaTeX).

    :param records: List of screen result dicts with keys {name, n_full, deep_iqm, shallow_iqm, delta, delta_lo, delta_hi, margin, verdict}.
    :param out_path: Path to write PNG to; PDF written to same basename with `.pdf` suffix.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted(records, key=lambda r: r["delta"])
    names = [r["name"] for r in recs]
    delta = np.array([r["delta"] for r in recs])
    lo = np.clip(delta - np.array([r["delta_lo"] for r in recs]), 0.0, None)
    hi = np.clip(np.array([r["delta_hi"] for r in recs]) - delta, 0.0, None)
    margin = recs[0]["margin"] if recs else 0.005

    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(recs) + 1.5))
    y = np.arange(len(recs))
    ax.errorbar(delta, y, xerr=[lo, hi], fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0.0, color="0.4", lw=1)
    ax.axvline(margin, color="0.7", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"$\Delta$ = IQM(deep) - IQM(shallow)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
