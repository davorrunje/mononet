# SPDX-License-Identifier: Apache-2.0
r"""Equal-budget tune + multi-seed IQM evaluation for one tier.

Produces the paper's headline (inverse) and mechanism-panel (forward) artifacts:
tune each method with an identical Optuna budget (once, seed 0), evaluate the best
config over ``--seeds`` seeds, and report the **interquartile mean** with a 95 %
bootstrap band — matching the repo's aggregation protocol
(:func:`benchmarks._common.results.interquartile_mean`). Writes a JSON artifact to
``results/`` for the manuscript to cite.

Example::

    uv run python -m applications.pinn.experiments.headline \\
        --problem lwr_riemann --tier inverse --out results/inverse-headline.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.run import run_one
from applications.pinn.experiments.search import search
from applications.pinn.models.protocol import ModelConfig

if TYPE_CHECKING:
    from applications.pinn.models.protocol import Backend, Method

_METHODS: tuple[Method, ...] = ("hard_monotone", "vanilla", "soft", "weight_clip")
Floats = npt.NDArray[np.float64]


def interquartile_mean(values: list[float]) -> float:
    """Mean of the middle 50 % (trim ``n//4`` from each end); matches the repo."""
    s = np.sort(np.asarray(values, dtype=np.float64))
    n = len(s)
    k = n // 4
    return float(s[k : n - k].mean()) if n - 2 * k > 0 else float(s.mean())


def bootstrap_band(
    values: list[float], rng: np.random.Generator, *, n_boot: int
) -> tuple[float, float]:
    """95 % percentile band of the IQM under seed resampling."""
    arr = np.asarray(values, dtype=np.float64)
    boots = [
        interquartile_mean(list(rng.choice(arr, len(arr), replace=True)))
        for _ in range(n_boot)
    ]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


_KEYS = ("l1", "l2", "admissibility_violation", "overshoot")


def run_panel(
    problem: str,
    tier: str,
    *,
    backend: Backend = "jax",
    n_trials: int = 20,
    seeds: int = 10,
    steps: int = 8000,
    grad_clip: float = 1.0,
    n_boot: int = 2000,
    residual: bool = True,
) -> dict[str, object]:
    """Tune + multi-seed IQM for all four methods on one ``(problem, tier)``.

    :param problem: Problem registry key.
    :param tier: ``"forward"`` or ``"inverse"``.
    :param backend: Backend to run on.
    :param n_trials: Equal Optuna budget per method.
    :param seeds: Number of evaluation seeds (IQM over these).
    :param steps: Optimisation steps per run.
    :param grad_clip: Global-norm gradient clip (stabilises the constrained field).
    :param n_boot: Bootstrap resamples for the band.
    :returns: A JSON-serialisable results dict.
    """
    rng = np.random.default_rng(0)
    tmpl = RunConfig(
        problem=problem,
        method="vanilla",
        backend=backend,
        tier=tier,
        steps=steps,
        grad_clip=grad_clip,
        model=ModelConfig(residual=residual),
    )
    out: dict[str, object] = {
        "problem": problem,
        "tier": tier,
        "backend": backend,
        "n_trials": n_trials,
        "steps": steps,
        "seeds": list(range(seeds)),
        "field": "residual" if residual else "plain",
        "aggregate": "iqm+bootstrap95",
        "methods": {},
    }
    methods_out: dict[str, object] = {}
    print(
        f"== {tier} panel: {problem}, {n_trials} trials tune + {seeds} seeds, "
        f"{steps} steps ==",
        flush=True,
    )
    for method in _METHODS:
        best = search(
            problem, method, backend, n_trials=n_trials, template=tmpl, seed=0
        )
        base = replace(
            tmpl,
            method=method,
            lr=best["lr"],
            residual_weight=best["residual_weight"],
            ic_weight=best.get("ic_weight", tmpl.ic_weight),
            data_weight=best.get("data_weight", tmpl.data_weight),
            soft_penalty=best.get("soft_penalty", tmpl.soft_penalty),
            model=replace(tmpl.model, width=int(best["width"])),
        )
        rows = [run_one(replace(base, seed=s)) for s in range(seeds)]
        agg = {}
        for key in _KEYS:
            vals = [float(r[key]) for r in rows]
            lo, hi = bootstrap_band(vals, rng, n_boot=n_boot)
            agg[key] = {"iqm": interquartile_mean(vals), "lo": lo, "hi": hi}
        methods_out[method] = {"best_params": best, "per_seed": rows, "agg": agg}
        a = agg
        print(
            f"{method:14} L1={a['l1']['iqm']:.3f} "
            f"[{a['l1']['lo']:.3f},{a['l1']['hi']:.3f}] "
            f"L2={a['l2']['iqm']:.3f} viol={a['admissibility_violation']['iqm']:.3f} "
            f"over={a['overshoot']['iqm']:.3f}",
            flush=True,
        )
    out["methods"] = methods_out
    return out


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--problem", default="lwr_riemann")
    p.add_argument("--tier", choices=["forward", "inverse"], default="inverse")
    p.add_argument("--backend", choices=["jax", "torch"], default="jax")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument(
        "--no-residual",
        action="store_true",
        help="Use a plain MonoLinear field instead of MonoResidual blocks.",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()
    result = run_panel(
        args.problem,
        args.tier,
        backend=args.backend,
        n_trials=args.n_trials,
        seeds=args.seeds,
        steps=args.steps,
        residual=not args.no_residual,
    )
    with Path(args.out).open("w") as f:
        json.dump(result, f, indent=2)
    print(f"== wrote {args.out} ==", flush=True)


if __name__ == "__main__":
    main()
