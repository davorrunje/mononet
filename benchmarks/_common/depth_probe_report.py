"""Δ = shallow_mse_iqm - deep_mse_iqm vs c from monotone-depth-probe records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.results import interquartile_mean

if TYPE_CHECKING:
    from pathlib import Path

_BOOT = 2000
_BOOT_SEED = 0

# Okabe-Ito colorblind-safe palette, cycled per `kind`.
_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")


def delta_by_c(
    records: list[dict[str, Any]], *, boot: int = _BOOT, seed: int = _BOOT_SEED
) -> list[dict[str, Any]]:
    """Per probe record: deep/shallow MSE IQMs, delta, and a bootstrap band.

    ``delta = shallow_mse_iqm - deep_mse_iqm``, so a positive delta means depth
    helps (lower MSE is better, unlike the accuracy-based size-ladder report).
    The band resamples ``deep_values`` and ``shallow_values`` independently
    (``boot`` draws, seeded ``np.random.default_rng(seed)``) and takes the
    2.5/97.5 percentiles of the bootstrapped IQM difference.

    :param records: Probe records with keys {kind, c, deep_mse_iqm,
        shallow_mse_iqm, deep_values, shallow_values} (see
        `benchmarks.monotone_depth_probe_run.probe_dataset`).
    :param boot: Number of bootstrap draws.
    :param seed: Seed for the bootstrap RNG.
    :returns: List of dicts with keys {kind, c, deep_mse_iqm, shallow_mse_iqm,
        delta, delta_lo, delta_hi}, in input order.
    """
    rng = np.random.default_rng(seed)
    out: list[dict[str, Any]] = []
    for r in records:
        dv = np.asarray(r["deep_values"], dtype=np.float64)
        sv = np.asarray(r["shallow_values"], dtype=np.float64)
        boot_delta = np.array(
            [
                interquartile_mean(rng.choice(sv, len(sv), replace=True))
                - interquartile_mean(rng.choice(dv, len(dv), replace=True))
                for _ in range(boot)
            ]
        )
        lo, hi = np.percentile(boot_delta, [2.5, 97.5])
        out.append(
            {
                "kind": r["kind"],
                "c": r["c"],
                "deep_mse_iqm": float(r["deep_mse_iqm"]),
                "shallow_mse_iqm": float(r["shallow_mse_iqm"]),
                "delta": float(r["shallow_mse_iqm"]) - float(r["deep_mse_iqm"]),
                "delta_lo": float(lo),
                "delta_hi": float(hi),
            }
        )
    return out


def probe_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table: one row per (kind, c) with deep/shallow MSE and Δ [CI].

    :param rows: Rows produced by :func:`delta_by_c`.
    :returns: Markdown-formatted table string.
    """
    head = "| kind | c | deep MSE | shallow MSE | Δ [95% CI] |\n|---|--:|--:|--:|--:|\n"
    body = [
        f"| {r['kind']} | {r['c']} | {r['deep_mse_iqm']:.4f} | "
        f"{r['shallow_mse_iqm']:.4f} | {r['delta']:+.4f} "
        f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}] |"
        for r in rows
    ]
    return head + "\n".join(body) + "\n"


def render_probe_plot(rows: list[dict[str, Any]], out_path: Path) -> None:
    r"""Render the Δ-vs-c figure next to `out_path`, as both PNG and PDF.

    One line per `kind`, with a dashed reference line at Δ=0 (no effect of
    depth). Writes ``out_path`` with a ``.png`` suffix (for the Sphinx/MD
    docs) and a ``.pdf`` suffix (vector, for ``\includegraphics`` in a LaTeX
    paper) from a single publication-styled render. No title is drawn —
    supply the docs heading / LaTeX caption instead.

    :param rows: Rows produced by :func:`delta_by_c`.
    :param out_path: Base output path; the suffix is replaced with png/pdf.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = sorted({r["kind"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(0.0, color="0.6", lw=1.0, ls="--")
    for kind, color in zip(kinds, _COLORS, strict=False):
        kind_rows = sorted((r for r in rows if r["kind"] == kind), key=lambda r: r["c"])
        cs = [r["c"] for r in kind_rows]
        delta = [r["delta"] for r in kind_rows]
        # Clip to >= 0: with few seeds the point IQM Δ can fall just outside
        # its own bootstrap band, which would make matplotlib reject a
        # negative error length.
        lo = np.clip([r["delta"] - r["delta_lo"] for r in kind_rows], 0.0, None)
        hi = np.clip([r["delta_hi"] - r["delta"] for r in kind_rows], 0.0, None)
        ax.errorbar(
            cs, delta, yerr=[lo, hi], marker="o", capsize=3, color=color, label=kind
        )
    ax.set_xlabel(r"synthetic complexity $c$", fontsize=12)
    ax.set_ylabel(r"$\Delta$ IQM MSE (shallow $-$ deep)", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)
