"""Depth probe: deep vs shallow monotone-residual on synthetic monotone targets.

For each (kind, c), run the standard search for the deep and shallow ``absolute``
residual arms on a synthetic monotone-regression bundle, refit + multi-seed test,
and record per-arm MSE IQM (+ raw per-seed values for the report's bootstrap).
See docs/superpowers/specs/2026-07-12-monotone-depth-synthetic-probe-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._common.results import interquartile_mean
from benchmarks._common.search import final_eval, search
from benchmarks.datasets.synthetic import synth_monotone


def _arm_mse(
    bundle: Any,
    *,
    deep: bool,
    n_trials: int,
    search_seeds: int,
    final_seeds: int,
    epochs: int,
    backend: str,
) -> list[float]:
    study = search(
        bundle,
        mode="absolute",
        residual=True,
        deep=deep,
        backend=backend,
        n_trials=n_trials,
        epochs=epochs,
        n_splits=1,
        search_seeds=search_seeds,
    )
    agg = final_eval(
        bundle,
        study.best_params,
        mode="absolute",
        residual=True,
        backend=backend,
        seeds=range(final_seeds),
        epochs=epochs,
    )
    return [float(v) for v in agg.values]


def probe_dataset(
    kind: str,
    c: int,
    *,
    n_trials: int = 15,
    search_seeds: int = 2,
    final_seeds: int = 8,
    epochs: int = 30,
    backend: str = "torch",
) -> dict[str, Any]:
    """Run both arms on ``synth_monotone(kind, c)``; return per-arm MSE IQMs + values.

    :param kind: Target family (``"additive"``, ``"teacher_relu"``,
        ``"teacher_elu"``, or ``"lattice"``).
    :param c: Complexity knob passed through to :func:`synth_monotone`.
    :param n_trials: Optuna trial budget per arm.
    :param search_seeds: Seeds-per-fold used inside the search objective.
    :param final_seeds: Number of refit-and-test seeds for the final MSE values.
    :param epochs: Training epochs, used for both search and final refit.
    :param backend: Backend passed through to ``search``/``final_eval``.
    :returns: Dict with ``kind``, ``c``, ``deep_mse_iqm``, ``shallow_mse_iqm``,
        ``deep_values``, and ``shallow_values`` (raw per-seed MSE lists).
    """
    bundle = synth_monotone(kind, c)  # type: ignore[arg-type]
    deep = _arm_mse(
        bundle,
        deep=True,
        n_trials=n_trials,
        search_seeds=search_seeds,
        final_seeds=final_seeds,
        epochs=epochs,
        backend=backend,
    )
    shallow = _arm_mse(
        bundle,
        deep=False,
        n_trials=n_trials,
        search_seeds=search_seeds,
        final_seeds=final_seeds,
        epochs=epochs,
        backend=backend,
    )
    return {
        "kind": kind,
        "c": c,
        "deep_mse_iqm": interquartile_mean(np.asarray(deep)),
        "shallow_mse_iqm": interquartile_mean(np.asarray(shallow)),
        "deep_values": deep,
        "shallow_values": shallow,
    }


def main() -> None:
    """CLI: sweep (kind, c) and write probe records JSON."""
    import argparse

    ap = argparse.ArgumentParser(description="monotone depth probe")
    ap.add_argument("--kinds", default="additive,teacher_relu,teacher_elu,lattice")
    ap.add_argument("--cs", default="1,2,4,8")
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--search-seeds", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [
        probe_dataset(
            k,
            int(c),
            n_trials=args.n_trials,
            search_seeds=args.search_seeds,
            final_seeds=args.final_seeds,
            epochs=args.epochs,
        )
        for k in args.kinds.split(",")
        for c in args.cs.split(",")
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recs, indent=2) + "\n")
    print(f"wrote {args.out} ({len(recs)} records)")  # noqa: T201


if __name__ == "__main__":
    main()
