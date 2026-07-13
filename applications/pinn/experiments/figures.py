# SPDX-License-Identifier: Apache-2.0
r"""Generate the paper figures from committed result artifacts + fresh reconstructions.

Two figure families:

1. **Crossover curves** (cheap, from the sweep JSONs): per-method L1/L2 error and
   admissibility violation vs observation noise, at a fixed observation count.
2. **Reconstruction profiles** (trains the tuned configs): a fixed-time spatial
   slice at a stress operating point, showing the hard-monotone field as a clean
   monotone ramp while the unconstrained/soft baselines oscillate across the shock,
   with the sparse noisy observations overlaid.

Writes PNGs to ``applications/pinn/paper/figures/``. Reconstructions need a JAX
backend (fast on ``gpu-jax``; fine on CPU for the six single-seed trainings).

Example::

    uv run python -m applications.pinn.experiments.figures
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from applications.pinn.core import plotting
from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.run import predict_field

if TYPE_CHECKING:
    from applications.pinn.models.protocol import Method

_FIG_DIR = Path("applications/pinn/paper/figures")
_RESULTS = Path("applications/pinn/results")
# Baselines shown in the paper (weight_clip is a diagnostic, omitted from figures).
_METHODS: tuple[Method, ...] = ("hard_monotone", "vanilla", "soft")
_LABELS = {
    "hard_monotone": "hard-monotone (ours)",
    "vanilla": "vanilla PINN",
    "soft": "soft-penalty",
}
# Stress operating point for the reconstruction money-shot.
_STRESS_N_OBS = 80
_STRESS_NOISE = 0.20


def _crossover(sweep_path: Path, key: str, ylabel: str, title: str, out: Path) -> None:
    """Build a per-method ``key``-vs-noise curve at ``_STRESS_N_OBS`` observations."""
    data = json.loads(sweep_path.read_text())
    noise = sorted({float(c["noise"]) for c in data["cells"]})
    series = {}
    for method in _METHODS:
        vals = {
            float(c["noise"]): float(c[key])
            for c in data["cells"]
            if c["method"] == method and int(c["n_obs"]) == _STRESS_N_OBS
        }
        series[_LABELS[method]] = np.array([vals[n] for n in noise])
    fig = plotting.metric_vs_noise(np.array(noise), series, ylabel=ylabel, title=title)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


def _reconstruction(
    problem: str, tuned_path: Path, pretty: str, out: Path, steps: int
) -> None:
    """Train each method at the stress point and plot a fixed-time spatial slice."""
    tuned = json.loads(tuned_path.read_text())
    series: dict[str, np.ndarray[Any, Any]] = {}
    x_values = t_values = None
    obs_slice: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None
    for method in _METHODS:
        best = tuned["methods"][method]["best_params"]
        cfg = RunConfig(
            problem=problem,
            method=method,
            backend="jax",
            tier="inverse",
            steps=steps,
            grad_clip=1.0,
            seed=0,
            n_obs=_STRESS_N_OBS,
            noise_std=_STRESS_NOISE,
            lr=float(best["lr"]),
            residual_weight=float(best["residual_weight"]),
            data_weight=float(best.get("data_weight", 1.0)),
            soft_penalty=float(best.get("soft_penalty", 0.0)),
        )
        cfg = replace(cfg, model=replace(cfg.model, width=int(best["width"])))
        r = predict_field(cfg)
        x_values, t_values = r.x_values, r.t_values
        # Fixed time slice near the end, where the shock is fully developed.
        ti = int(0.85 * (len(t_values) - 1))
        if "true" not in series:
            series["true"] = r.ref[ti]
        series[_LABELS[method]] = r.pred[ti]
        if obs_slice is None and r.obs is not None:
            oc, ov = r.obs
            band = 0.12 * float(t_values[-1] - t_values[0])
            near = np.abs(oc[:, 1] - float(t_values[ti])) <= band
            obs_slice = (oc[near, 0], ov[near])
    assert x_values is not None
    assert t_values is not None
    ti = int(0.85 * (len(t_values) - 1))
    fig = plotting.reconstruction_profile(
        x_values,
        series,
        obs=obs_slice,
        title=(
            f"{pretty}: inverse reconstruction at t={float(t_values[ti]):.2f}\n"
            f"(n_obs={_STRESS_N_OBS}, noise={_STRESS_NOISE})"
        ),
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}", flush=True)


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument(
        "--no-reconstructions",
        action="store_true",
        help="Only rebuild crossover curves from JSONs (no training).",
    )
    args = p.parse_args()
    _FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Crossover curves (from committed sweeps) -- LWR and Burgers, L1 and L2.
    panels = [
        ("inverse-sweep.json", "LWR (traffic, concave flux)", "lwr"),
        ("inverse-burgers-sweep.json", "Burgers (convex flux)", "burgers"),
    ]
    for fname, pretty, tag in panels:
        path = _RESULTS / fname
        if not path.exists():
            print(f"skip crossover: {path} missing", flush=True)
            continue
        _crossover(
            path,
            "l1",
            "L1 error (whole field)",
            f"{pretty}: whole-field L1 vs noise (n_obs={_STRESS_N_OBS})",
            _FIG_DIR / f"crossover-{tag}-l1.png",
        )
        _crossover(
            path,
            "l2",
            "L2 error (front-weighted)",
            f"{pretty}: front-weighted L2 vs noise (n_obs={_STRESS_N_OBS})",
            _FIG_DIR / f"crossover-{tag}-l2.png",
        )
        _crossover(
            path,
            "admissibility_violation",
            "admissibility violation",
            f"{pretty}: admissibility violation vs noise (n_obs={_STRESS_N_OBS})",
            _FIG_DIR / f"admissibility-{tag}.png",
        )

    if args.no_reconstructions:
        return
    recon = [
        ("lwr_riemann", "inverse-headline.json", "LWR (traffic)", "lwr"),
        ("burgers_riemann", "inverse-burgers-headline.json", "Burgers", "burgers"),
    ]
    for problem, tuned, pretty, tag in recon:
        tpath = _RESULTS / tuned
        if not tpath.exists():
            print(f"skip reconstruction: {tpath} missing", flush=True)
            continue
        _reconstruction(
            problem, tpath, pretty, _FIG_DIR / f"reconstruction-{tag}.png", args.steps
        )


if __name__ == "__main__":
    main()
