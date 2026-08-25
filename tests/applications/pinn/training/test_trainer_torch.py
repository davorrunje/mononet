# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the PyTorch PINN trainer."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from applications.pinn.core import sampling
from applications.pinn.core.admissibility import violation
from applications.pinn.core.problems import conservation
from applications.pinn.models.protocol import ModelConfig
from applications.pinn.models.torch import builders
from applications.pinn.training import torch_trainer
from applications.pinn.training.losses import LossWeights, TrainingData

CFG = ModelConfig(width=16, n_blocks=2, t_embed_dim=4, seed=0)


def _burgers_forward_data() -> tuple[conservation.BurgersRiemann, TrainingData]:
    problem = conservation.BurgersRiemann(u_l=1.0, u_r=0.0)
    domain = problem.domain
    coll = sampling.collocation(domain, 256, seed=0)
    ic_pts = sampling.initial_points(domain, 64, seed=1)
    ic_vals = problem.initial(ic_pts[:, 0])
    return problem, TrainingData(collocation=coll, ic=(ic_pts, ic_vals))


def test_training_reduces_loss() -> None:
    """A few Adam steps reduce the PINN loss (residual autodiff works)."""
    problem, data = _burgers_forward_data()
    model = builders.build_torch(problem, CFG, "hard_monotone")
    _, history = torch_trainer.train(
        problem, model, data, weights=LossWeights(), sign_x=-1, lr=3e-3, steps=60
    )
    assert history[-1] < history[0]
    assert np.isfinite(history[-1])


def test_hard_monotone_stays_admissible_after_training() -> None:
    """The trained hard-monotone field is still non-increasing in x (violation 0)."""
    problem, data = _burgers_forward_data()
    model = builders.build_torch(problem, CFG, "hard_monotone")
    trained, _ = torch_trainer.train(
        problem, model, data, weights=LossWeights(), sign_x=-1, lr=3e-3, steps=40
    )
    x_values, _ = sampling.eval_grid(problem.domain, 60, 3)
    x = torch.as_tensor(x_values, dtype=torch.float32).reshape(-1, 1)
    t = torch.full_like(x, 0.5)
    u = trained(x, t).detach().numpy().ravel()
    assert violation(u, axis=0, sign=-1) < 1e-5


def test_vanilla_also_trains() -> None:
    """The unconstrained baseline also runs and reduces its loss."""
    problem, data = _burgers_forward_data()
    model = builders.build_torch(problem, CFG, "vanilla")
    _, history = torch_trainer.train(
        problem, model, data, weights=LossWeights(), sign_x=-1, lr=3e-3, steps=40
    )
    assert history[-1] < history[0]
