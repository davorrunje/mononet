# SPDX-License-Identifier: Apache-2.0
r"""Inverse-tier robustness sweep: reconstruction quality vs sparsity x noise.

Each method is tuned **once** at the reference operating point (loaded from
``results/inverse-headline.json``), then **stress-tested** across a grid of
observation counts and noise levels — mirroring deployment, where you tune once
and face varying data quality. Per cell we report the multi-seed IQM of L² error,
monotonicity violation, near-shock overshoot, and the **out-of-range fraction**
(a physical-validity proxy: predictions the true field cannot contain).

The thesis this probes (opportunity #2): as data thins / noise grows, the
unconstrained/soft baselines should oscillate and produce unphysical values, while
the hard-monotone field stays admissible by construction.

Example::

    uv run python -m applications.pinn.experiments.sweep_inverse \
        --tuned results/inverse-headline.json --out results/inverse-sweep.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.headline import interquartile_mean
from applications.pinn.experiments.run import run_one

if TYPE_CHECKING:
    from applications.pinn.models.protocol import Method

_METHODS: tuple[Method, ...] = ("hard_monotone", "vanilla", "soft", "weight_clip")
_N_OBS = (20, 40, 80, 160)
_NOISE = (0.0, 0.05, 0.1)
_KEYS = ("l2", "admissibility_violation", "overshoot", "oob_frac")


def _base_config(
    problem: str, method: Method, best: dict[str, Any], steps: int
) -> RunConfig:
    """Rebuild a tuned RunConfig for ``method`` from its stored best params."""
    tmpl = RunConfig(
        problem=problem,
        method=method,
        backend="jax",
        tier="inverse",
        steps=steps,
        grad_clip=1.0,
    )
    return replace(
        tmpl,
        lr=float(best["lr"]),
        residual_weight=float(best["residual_weight"]),
        data_weight=float(best.get("data_weight", tmpl.data_weight)),
        soft_penalty=float(best.get("soft_penalty", tmpl.soft_penalty)),
        model=replace(tmpl.model, width=int(best["width"])),
    )


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tuned", default="applications/pinn/results/inverse-headline.json")
    p.add_argument("--out", default="applications/pinn/results/inverse-sweep.json")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=6000)
    args = p.parse_args()

    tuned = json.loads(Path(args.tuned).read_text())
    problem = str(tuned["problem"])
    print(
        f"== inverse sweep: {problem}, {len(_N_OBS)}x{len(_NOISE)} cells, "
        f"{args.seeds} seeds, {args.steps} steps ==",
        flush=True,
    )

    out: dict[str, Any] = {
        "problem": problem,
        "n_obs": list(_N_OBS),
        "noise": list(_NOISE),
        "seeds": args.seeds,
        "steps": args.steps,
        "aggregate": "iqm",
        "cells": [],
    }
    for method in _METHODS:
        best = tuned["methods"][method]["best_params"]
        base = _base_config(problem, method, best, args.steps)
        for n_obs in _N_OBS:
            for noise in _NOISE:
                rows = [
                    run_one(replace(base, seed=s, n_obs=n_obs, noise_std=noise))
                    for s in range(args.seeds)
                ]
                agg = {
                    k: interquartile_mean([float(r[k]) for r in rows]) for k in _KEYS
                }
                out["cells"].append(
                    {"method": method, "n_obs": n_obs, "noise": noise, **agg}
                )
                print(
                    f"{method:14} n_obs={n_obs:4d} noise={noise:.2f} | "
                    f"L2={agg['l2']:.3f} viol={agg['admissibility_violation']:.3f} "
                    f"over={agg['overshoot']:.3f} oob={agg['oob_frac']:.3f}",
                    flush=True,
                )

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"== wrote {args.out} ==", flush=True)


if __name__ == "__main__":
    main()
