# SPDX-License-Identifier: Apache-2.0
"""Configuration for a single PINN experiment run.

A ``RunConfig`` fully determines one ``(problem, method, backend, seed)`` run:
architecture, point counts, optimisation, loss weights, and the evaluation grid.
It is JSON-round-trippable (stdlib dataclasses) for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from applications.pinn.models.protocol import Backend, Method, ModelConfig


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Everything needed to run and score one experiment.

    :param problem: Registry key of the problem.
    :param method: One of the four methods.
    :param backend: ``"torch"`` or ``"jax"``.
    :param seed: Seed for model init and point sampling.
    :param model: Architecture configuration.
    :param n_collocation: Interior collocation points.
    :param n_ic: Initial-condition points (forward tier).
    :param n_bc: Boundary points (forward tier).
    :param steps: Optimisation steps.
    :param lr: Adam learning rate.
    :param residual_weight: Residual loss weight.
    :param ic_weight: IC loss weight.
    :param bc_weight: BC loss weight.
    :param soft_penalty: Monotonicity-penalty weight for the ``soft`` method
        (ignored otherwise; the whole point is that only ``soft`` uses it).
    :param eval_nx: Evaluation-grid spatial resolution.
    :param eval_nt: Evaluation-grid temporal resolution.
    """

    problem: str
    method: Method
    backend: Backend
    seed: int = 0
    model: ModelConfig = field(default_factory=ModelConfig)
    n_collocation: int = 2000
    n_ic: int = 256
    n_bc: int = 128
    steps: int = 2000
    lr: float = 1e-3
    residual_weight: float = 1.0
    ic_weight: float = 10.0
    bc_weight: float = 1.0
    soft_penalty: float = 1.0
    eval_nx: int = 200
    eval_nt: int = 50
    #: ``"forward"`` (IC/BC data) or ``"inverse"`` (sparse observations + data-fit).
    tier: str = "forward"
    #: Inverse tier: number of sparse observations drawn from the reference field.
    n_obs: int = 80
    #: Inverse tier: Gaussian noise std added to observations.
    noise_std: float = 0.02
    #: Inverse tier: weight on the observation data-fit term.
    data_weight: float = 10.0
    #: Optax global-norm gradient clip (stabilises the constrained-field residual).
    grad_clip: float = 1.0
