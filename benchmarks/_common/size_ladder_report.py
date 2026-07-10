"""Δ = IQM(deep) - IQM(shallow) vs N from a size-ladder results JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.results import interquartile_mean

if TYPE_CHECKING:
    from pathlib import Path

_BOOT = 2000
_BOOT_SEED = 0


def delta_by_n(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per N: deep_iqm, shallow_iqm, delta, and a bootstrap percentile band.

    The band resamples the deep and shallow per-seed value vectors independently
    (`_BOOT` draws, seed `_BOOT_SEED`) and takes the 2.5/97.5 percentiles of the
    bootstrapped IQM difference.
    """
    rng = np.random.default_rng(_BOOT_SEED)
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
        boot = np.array(
            [
                interquartile_mean(rng.choice(dv, len(dv), replace=True))
                - interquartile_mean(rng.choice(sv, len(sv), replace=True))
                for _ in range(_BOOT)
            ]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(
            {
                "n": n,
                "deep_iqm": float(deep["test_iqm"]),
                "shallow_iqm": float(shallow["test_iqm"]),
                "delta": float(deep["test_iqm"]) - float(shallow["test_iqm"]),
                "delta_lo": float(lo),
                "delta_hi": float(hi),
            }
        )
    return out


def render_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    """Render Δ-vs-N (log-N x-axis) with the bootstrap band to `out_path` (PNG)."""
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
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="0.7", lw=1)
    ax.errorbar(ns, delta, yerr=[lo, hi], marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train size N (log)")
    ax.set_ylabel("Δ IQM  (deep - shallow)")
    ax.set_title("loan: deep-vs-shallow accuracy gap vs training size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
