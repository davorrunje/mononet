# SPDX-License-Identifier: Apache-2.0
r"""Non-PINN baseline: a classical smoother for the inverse reconstruction.

Isolates *monotonicity* from *generic smoothing* (and from the PINN itself). We
fit a thin-plate-spline RBF smoother **directly to the sparse noisy observations**
(no PDE residual, no monotonicity), with the smoothing strength tuned by
held-out-observation cross-validation (data-only, as at deployment). If the
hard-monotone PINN beats this tuned smoother as noise grows, the win is specific to
expressive hard monotonicity, not smoothing per se.

CPU-only (scipy). Reports the same L2 / monotonicity-violation / overshoot as the
sweep, multi-seed IQM, across the observation-count x noise grid.

Example::

    uv run python -m applications.pinn.experiments.baselines \
        --out applications/pinn/results/inverse-baseline-smoother.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RBFInterpolator

from applications.pinn.core import metrics, sampling
from applications.pinn.core.admissibility import violation
from applications.pinn.core.problems import get
from applications.pinn.experiments.headline import interquartile_mean

Array = npt.NDArray[np.float64]
_N_OBS = (20, 40, 80, 160)
_NOISE = (0.0, 0.05, 0.1, 0.15, 0.2)
_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _norm(coords: Array, bounds: tuple[float, float, float, float]) -> Array:
    """Map (x, t) columns to [-1, 1] (RBF is scale-sensitive)."""
    x0, x1, t0, t1 = bounds
    out = np.empty_like(coords)
    out[:, 0] = 2.0 * (coords[:, 0] - x0) / (x1 - x0) - 1.0
    out[:, 1] = 2.0 * (coords[:, 1] - t0) / (t1 - t0) - 1.0
    return out


def _fit_predict(
    obs_coords: Array, obs_vals: Array, grid: Array, smoothing: float
) -> Array:
    """Thin-plate-spline RBF smoother fit to observations, evaluated on ``grid``."""
    model = RBFInterpolator(
        obs_coords, obs_vals, kernel="thin_plate_spline", smoothing=smoothing
    )
    return np.asarray(model(grid), dtype=np.float64)


def _best_lambda(obs_coords: Array, obs_vals: Array, seed: int) -> float:
    """Tune smoothing by a 75/25 held-out-observation split (data-only)."""
    rng = np.random.default_rng(seed)
    n = len(obs_vals)
    perm = rng.permutation(n)
    k = max(1, n // 4)
    val, fit = perm[:k], perm[k:]
    best_lam, best_err = _LAMBDAS[0], np.inf
    for lam in _LAMBDAS:
        pred = _fit_predict(obs_coords[fit], obs_vals[fit], obs_coords[val], lam)
        err = float(np.mean((pred - obs_vals[val]) ** 2))
        if err < best_err:
            best_err, best_lam = err, lam
    return best_lam


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="lwr_riemann")
    p.add_argument(
        "--out", default="applications/pinn/results/inverse-baseline-smoother.json"
    )
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--eval-nx", type=int, default=200)
    p.add_argument("--eval-nt", type=int, default=60)
    args = p.parse_args()

    prob = get(args.problem)()
    (x0, x1), (t0, t1) = prob.domain
    bounds = (float(x0), float(x1), float(t0), float(t1))
    sign_x = int(prob.admissibility().mask[0])
    xs, ts = sampling.eval_grid(prob.domain, args.eval_nx, args.eval_nt)
    gx, gt = np.meshgrid(xs, ts)
    ref_raw = prob.ground_truth(gx.ravel(), gt.ravel())
    assert ref_raw is not None
    ref = np.asarray(ref_raw, dtype=np.float64).reshape(args.eval_nt, args.eval_nx)
    grid = _norm(np.column_stack([gx.ravel(), gt.ravel()]), bounds)
    dx = float(xs[1] - xs[0])

    print(
        f"== non-PINN smoother baseline: {args.problem}, "
        f"{len(_N_OBS)}x{len(_NOISE)} cells, {args.seeds} seeds ==",
        flush=True,
    )
    out: dict[str, Any] = {
        "problem": args.problem,
        "method": "rbf_smoother",
        "n_obs": list(_N_OBS),
        "noise": list(_NOISE),
        "seeds": args.seeds,
        "aggregate": "iqm",
        "cells": [],
    }
    for n_obs in _N_OBS:
        for noise in _NOISE:
            l1s, l2s, viols, overs = [], [], [], []
            for seed in range(args.seeds):
                oc_raw, ov_raw = sampling.observations(
                    ref, xs, ts, n_obs=n_obs, noise_std=noise, seed=seed + 3
                )
                oc = np.asarray(oc_raw, dtype=np.float64)
                ov = np.asarray(ov_raw, dtype=np.float64)
                ocn = _norm(oc, bounds)
                lam = _best_lambda(ocn, ov, seed)
                pred = _fit_predict(ocn, ov, grid, lam).reshape(
                    args.eval_nt, args.eval_nx
                )
                l1s.append(metrics.l1(pred, ref, dx=dx))
                l2s.append(metrics.l2(pred, ref, dx=dx))
                viols.append(
                    max(
                        violation(pred[i], axis=0, sign=sign_x)
                        for i in range(args.eval_nt)
                    )
                )
                overs.append(
                    max(metrics.overshoot(pred[i], ref[i]) for i in range(args.eval_nt))
                )
            cell = {
                "method": "rbf_smoother",
                "n_obs": n_obs,
                "noise": noise,
                "l1": interquartile_mean(l1s),
                "l2": interquartile_mean(l2s),
                "admissibility_violation": interquartile_mean(viols),
                "overshoot": interquartile_mean(overs),
            }
            out["cells"].append(cell)
            print(
                f"rbf_smoother n_obs={n_obs:4d} noise={noise:.2f} | "
                f"L1={cell['l1']:.3f} L2={cell['l2']:.3f} "
                f"viol={cell['admissibility_violation']:.3f} "
                f"over={cell['overshoot']:.3f}",
                flush=True,
            )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"== wrote {args.out} ==", flush=True)


if __name__ == "__main__":
    main()
