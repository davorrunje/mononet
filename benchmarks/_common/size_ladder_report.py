"""Δ = IQM(deep) - IQM(shallow) vs N from a size-ladder results JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.results import bootstrap_delta

if TYPE_CHECKING:
    from pathlib import Path

_BOOT = 2000
_BOOT_SEED = 0


def delta_by_n(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per N: deep_iqm, shallow_iqm, delta, and a bootstrap percentile band.

    The band resamples the deep and shallow per-seed value vectors independently
    (`_BOOT` draws, seed `_BOOT_SEED`) and takes the 2.5/97.5 percentiles of the
    bootstrapped IQM difference, via the shared
    :func:`benchmarks._common.results.bootstrap_delta`. This report is used for
    accuracy-style (higher-is-better) metrics only, so ``lower_is_better`` is
    fixed at ``False``.
    """
    by_n: dict[int, dict[str, dict[str, Any]]] = {}
    for r in records:
        by_n.setdefault(int(r["n"]), {})[r["arm"]] = r
    out: list[dict[str, Any]] = []
    for n in sorted(by_n):
        deep, shallow = by_n[n].get("deep"), by_n[n].get("shallow")
        if deep is None or shallow is None:
            continue
        dv = np.asarray(deep["test_values"], dtype=np.float64)
        sv = np.asarray(shallow["test_values"], dtype=np.float64)
        point, lo, hi = bootstrap_delta(
            dv, sv, lower_is_better=False, n_boot=_BOOT, seed=_BOOT_SEED
        )
        out.append(
            {
                "n": n,
                "deep_iqm": float(deep["test_iqm"]),
                "shallow_iqm": float(shallow["test_iqm"]),
                "delta": point,
                "delta_lo": lo,
                "delta_hi": hi,
            }
        )
    return out


def render_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    r"""Render the Δ-vs-N figure next to `out_path`, as both PNG and PDF.

    Writes ``out_path`` with a ``.png`` suffix (for the Sphinx/MD docs) and a
    ``.pdf`` suffix (vector, for ``\includegraphics`` in a LaTeX paper) from a
    single publication-styled render. Labels use matplotlib mathtext, so no
    LaTeX toolchain is required. No title is drawn — supply the docs heading /
    LaTeX caption instead.

    :param records: Size-ladder result records (see :func:`delta_by_n`).
    :param out_path: Base output path; the suffix is replaced with png/pdf.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = delta_by_n(records)
    ns = [r["n"] for r in rows]
    delta = [r["delta"] for r in rows]
    # Clip to >= 0: with few seeds the point IQM Δ can fall just outside its own
    # bootstrap band, which would make matplotlib reject a negative error length.
    lo = np.clip([r["delta"] - r["delta_lo"] for r in rows], 0.0, None)
    hi = np.clip([r["delta_hi"] - r["delta"] for r in rows], 0.0, None)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.axhline(0.0, color="0.6", lw=1.0, ls="--")
    ax.errorbar(ns, delta, yerr=[lo, hi], marker="o", capsize=3, color="#0072B2")
    ax.set_xscale("log")
    ax.set_xlabel(r"training-set size $N$", fontsize=12)
    ax.set_ylabel(r"$\Delta$ IQM accuracy (deep $-$ shallow)", fontsize=12)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)
