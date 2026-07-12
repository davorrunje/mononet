# SPDX-License-Identifier: Apache-2.0
"""PyTorch (Adam) training loop for the PINN methods.

Mirrors the JAX trainer. Input derivatives ``u_x, u_t`` for the PDE residual come
from ``torch.autograd.grad`` with ``create_graph=True`` so the residual stays in
the graph and its parameter gradients flow. First-order scalar conservation laws
need only first derivatives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from applications.pinn.training.losses import LossWeights, TrainingData


def _col(values: object) -> torch.Tensor:
    """Return a float32 column tensor from a NumPy array."""
    return torch.as_tensor(values, dtype=torch.float32).reshape(-1, 1)


def train(
    problem: object,
    model: nn.Module,
    data: TrainingData,
    *,
    weights: LossWeights,
    sign_x: int,
    lr: float = 1e-3,
    steps: int = 200,
) -> tuple[nn.Module, list[float]]:
    """Train ``model`` with Adam on the PINN loss; return it and the loss history.

    :param problem: Registered problem (supplies ``flux_prime`` for the residual).
    :param model: A torch model callable as ``u(x, t)``.
    :param data: Collocation and (IC/BC or observation) point sets.
    :param weights: Loss-term weights.
    :param sign_x: Desired monotonicity sign in ``x`` (for the soft penalty).
    :param lr: Adam learning rate.
    :param steps: Number of optimisation steps.
    :returns: ``(trained_model, loss_history)``.
    """
    xc = _col(data.collocation[:, 0]).requires_grad_(True)
    tc = _col(data.collocation[:, 1]).requires_grad_(True)
    supervised = [
        (_col(pair[0][:, 0]), _col(pair[0][:, 1]), _col(pair[1]), w)
        for pair, w in (
            (data.ic, weights.ic),
            (data.bc, weights.bc),
            (data.obs, weights.data),
        )
        if pair is not None
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad()
        u = model(xc, tc)
        (u_x,) = torch.autograd.grad(u.sum(), xc, create_graph=True)
        (u_t,) = torch.autograd.grad(u.sum(), tc, create_graph=True)
        residual = u_t + problem.flux_prime(u) * u_x  # type: ignore[attr-defined]
        loss = weights.residual * (residual**2).mean()
        if weights.mono > 0.0:
            loss = loss + weights.mono * torch.relu(-sign_x * u_x).pow(2).mean()
        for cx, ct, cv, w in supervised:
            loss = loss + w * (model(cx, ct) - cv).pow(2).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))

    return model, history
