# SPDX-License-Identifier: Apache-2.0
"""Run and score a single PINN experiment.

Orchestrates: build model (backend + method) -> assemble training data ->
train -> predict on the evaluation grid -> score against ground truth. Returns a
plain-dict artifact (JSON-serialisable) with the headline metrics, including the
admissibility violation that is ``0`` by construction for the hard-monotone model.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from applications.pinn.core import metrics, sampling
from applications.pinn.core.admissibility import violation
from applications.pinn.core.problems import get
from applications.pinn.models.protocol import ModelConfig, build
from applications.pinn.training.losses import LossWeights, TrainingData

if TYPE_CHECKING:
    from applications.pinn.experiments.config import RunConfig

Array = npt.NDArray[np.floating]


def _ground_truth(problem: object, x: Array, t: Array) -> Array:
    """Evaluate a problem's ground truth, requiring it to be available."""
    value = problem.ground_truth(x, t)  # type: ignore[attr-defined]
    if value is None:
        raise ValueError("problem has no ground truth for scoring")
    result: Array = value
    return result


def _predict(model: object, coords: Array, backend: str) -> Array:
    """Evaluate a trained model on ``coords`` (N, 2), returning a NumPy vector."""
    x = coords[:, 0:1]
    t = coords[:, 1:2]
    if backend == "torch":
        import torch

        xt = torch.as_tensor(x, dtype=torch.float32)
        tt = torch.as_tensor(t, dtype=torch.float32)
        return model(xt, tt).detach().numpy().ravel()  # type: ignore[operator,no-any-return]
    import jax.numpy as jnp

    out = model(jnp.asarray(x), jnp.asarray(t))  # type: ignore[operator]
    return np.asarray(out).ravel()


def _train(
    problem: object, model: object, data: TrainingData, cfg: RunConfig
) -> object:
    mono = cfg.soft_penalty if cfg.method == "soft" else 0.0
    if cfg.tier == "inverse":
        # Inverse: fit sparse observations + PDE residual, no IC/BC.
        weights = LossWeights(
            residual=cfg.residual_weight,
            ic=0.0,
            bc=0.0,
            data=cfg.data_weight,
            mono=mono,
        )
    else:
        weights = LossWeights(
            residual=cfg.residual_weight,
            ic=cfg.ic_weight,
            bc=cfg.bc_weight,
            data=0.0,
            mono=mono,
        )
    sign_x = int(problem.admissibility().mask[0])  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {
        "weights": weights,
        "sign_x": sign_x,
        "lr": cfg.lr,
        "steps": cfg.steps,
        "grad_clip": cfg.grad_clip,
    }
    if cfg.backend == "torch":
        from applications.pinn.training import torch_trainer

        trained, _ = torch_trainer.train(problem, model, data, **kwargs)  # type: ignore[arg-type]
        return trained
    from applications.pinn.training import jax_trainer

    trained_jax, _ = jax_trainer.train(problem, model, data, **kwargs)  # type: ignore[arg-type]
    return trained_jax


def run_one(cfg: RunConfig) -> dict[str, Any]:
    """Run one experiment and return its metrics artifact.

    :param cfg: The fully-specified run configuration.
    :returns: A JSON-serialisable dict of configuration + headline metrics.
    """
    problem = get(cfg.problem)()
    domain = problem.domain
    model = build(problem, cfg.model, cfg.method, cfg.backend)

    x_values, t_values = sampling.eval_grid(domain, cfg.eval_nx, cfg.eval_nt)
    collocation = sampling.collocation(domain, cfg.n_collocation, seed=cfg.seed)
    if cfg.tier == "inverse":
        # Reconstruct from sparse noisy observations of the reference field.
        ref_field = _ground_truth(
            problem, *(a.ravel() for a in np.meshgrid(x_values, t_values))
        ).reshape(cfg.eval_nt, cfg.eval_nx)
        obs_coords, obs_vals = sampling.observations(
            ref_field,
            x_values,
            t_values,
            n_obs=cfg.n_obs,
            noise_std=cfg.noise_std,
            seed=cfg.seed + 3,
        )
        data = TrainingData(collocation=collocation, obs=(obs_coords, obs_vals))
    else:
        ic_pts = sampling.initial_points(domain, cfg.n_ic, seed=cfg.seed + 1)
        ic_vals = problem.initial(ic_pts[:, 0])  # type: ignore[attr-defined]
        bc_pts = sampling.boundary_points(domain, cfg.n_bc, seed=cfg.seed + 2)
        bc_vals = _ground_truth(problem, bc_pts[:, 0], bc_pts[:, 1])
        data = TrainingData(
            collocation=collocation, ic=(ic_pts, ic_vals), bc=(bc_pts, bc_vals)
        )

    trained = _train(problem, model, data, cfg)

    grid_x, grid_t = np.meshgrid(x_values, t_values)
    coords = np.column_stack([grid_x.ravel(), grid_t.ravel()])
    pred = _predict(trained, coords, cfg.backend).reshape(cfg.eval_nt, cfg.eval_nx)
    ref = _ground_truth(problem, grid_x.ravel(), grid_t.ravel()).reshape(
        cfg.eval_nt, cfg.eval_nx
    )

    sign_x = int(problem.admissibility().mask[0])
    dx = float(x_values[1] - x_values[0])
    viol = max(violation(pred[i], axis=0, sign=sign_x) for i in range(cfg.eval_nt))
    over = max(metrics.overshoot(pred[i], ref[i]) for i in range(cfg.eval_nt))
    # Physical-validity proxy: fraction of predictions outside the true field's
    # range — i.e. unphysical over/undershoot the reference cannot contain.
    lo, hi = float(ref.min()), float(ref.max())
    oob_frac = float(np.mean((pred < lo) | (pred > hi)))
    return {
        "problem": cfg.problem,
        "method": cfg.method,
        "backend": cfg.backend,
        "seed": cfg.seed,
        "l1": metrics.l1(pred, ref, dx=dx),
        "l2": metrics.l2(pred, ref, dx=dx),
        "admissibility_violation": viol,
        "overshoot": over,
        "oob_frac": oob_frac,
    }


def main() -> None:
    """CLI: run one experiment and print its metrics artifact as JSON."""
    import json

    from applications.pinn.experiments.config import RunConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem")
    parser.add_argument("method")
    parser.add_argument("backend", choices=["torch", "jax"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()
    cfg = RunConfig(
        problem=args.problem,
        method=args.method,
        backend=args.backend,
        seed=args.seed,
        steps=args.steps,
        model=ModelConfig(),
    )
    print(json.dumps(run_one(cfg), indent=2))


if __name__ == "__main__":
    main()
