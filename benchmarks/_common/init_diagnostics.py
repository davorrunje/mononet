"""Diagnostics for mixed-vs-split init conditioning across depth (torch)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mononet.torch import MonoLinear, MonoResidual

if TYPE_CHECKING:
    from mononet.core.config import Mode
    from mononet.core.types import ActivationName


def synthetic_monotone(n: int, d: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardized features with a known monotone target.

    :param n: Number of samples.
    :param d: Number of features.
    :param seed: RNG seed.
    :returns: ``(X, y)`` with ``X`` standardized ``(n, d)`` and ``y`` standardized ``(n,)``;
        ``y = Σ softplus(aᵢ·xᵢ) + ε``, ``aᵢ > 0`` (non-decreasing in every feature).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    a = rng.uniform(0.5, 1.5, size=d)
    y = np.logaddexp(0.0, x * a).sum(axis=1) + 0.05 * rng.standard_normal(n)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return x, y


def _stack(
    mode: Mode, depth: int, d: int, width: int, activation: ActivationName
) -> nn.Module:
    layers: list[nn.Module] = [MonoLinear(d, width, mode=mode, activation=activation)]
    layers += [
        MonoLinear(width, width, mode=mode, activation=activation)
        for _ in range(depth - 1)
    ]
    layers.append(MonoLinear(width, 1, mode=mode, activation=activation))
    return nn.Sequential(*[layer.double() for layer in layers])


def grad_flow(
    mode: Mode,
    depth: int,
    *,
    activation: ActivationName = "elu",
    width: int = 32,
    seed: int = 0,
) -> dict[str, float | list[float]]:
    """Init-time gradient flow through an untrained plain stack.

    :param mode: ``split`` or ``mixed``.
    :param depth: Number of hidden ``MonoLinear`` layers.
    :param activation: Base activation.
    :param width: Hidden width.
    :param seed: RNG seed.
    :returns: ``{"input_grad_norm": float, "layer_grad_norms": [float, ...]}`` (len == depth).
    """
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    net = _stack(mode, depth, x_np.shape[1], width, activation)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    loss = nn.functional.mse_loss(net(x), y)
    loss.backward()  # type: ignore[no-untyped-call]
    assert x.grad is not None
    hidden = [m for m in net.children() if isinstance(m, MonoLinear)][:depth]
    layer_grad_norms: list[float] = []
    for m in hidden:
        assert m.weight.grad is not None
        layer_grad_norms.append(float(m.weight.grad.norm()))
    return {
        "input_grad_norm": float(x.grad.norm()),
        "layer_grad_norms": layer_grad_norms,
    }


def trainability(
    mode: Mode,
    depth: int,
    *,
    activation: ActivationName = "elu",
    epochs: int = 100,
    seed: int = 0,
) -> dict[str, float]:
    """Fixed-budget train loss of a plain stack on the synthetic target.

    :param mode: ``split`` or ``mixed``.
    :param depth: Number of hidden ``MonoLinear`` layers.
    :param activation: Base activation.
    :param epochs: Full-batch training epochs.
    :param seed: RNG seed.
    :returns: ``{"final_train_loss": float, "epochs_to_threshold": float}`` (threshold 0.5 MSE;
        ``inf`` if never reached).
    """
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    net = _stack(mode, depth, x_np.shape[1], 32, activation)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    hit = float("inf")
    loss_val = float("inf")
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
        if loss_val < 0.5 and hit == float("inf"):
            hit = float(ep)
    return {"final_train_loss": loss_val, "epochs_to_threshold": hit}


def build_residual_stack(
    mode: Mode, depth: int, sub_depth: int | None, *, width: int = 32
) -> nn.Module:
    """Uniform-width monotone stack; residual (skip every ``sub_depth``) or plain.

    :param mode: ``split`` or ``mixed``.
    :param depth: Number of hidden ``W->W`` monotone layers.
    :param sub_depth: Layers per residual block; ``None`` builds a plain (no-skip) stack.
    :param width: Uniform hidden width ``W``.
    :returns: An ``nn.Sequential`` mapping ``(batch, 8) -> (batch, 1)``.
    """
    layers: list[nn.Module] = [MonoLinear(8, width, mode=mode, activation="elu")]
    if sub_depth is None:
        layers += [
            MonoLinear(width, width, mode=mode, activation="elu") for _ in range(depth)
        ]
    else:
        layers += [
            MonoResidual(width, width, mode=mode, activation="elu", sub_depth=sub_depth)
            for _ in range(depth // sub_depth)
        ]
    layers.append(MonoLinear(width, 1, mode=mode, activation="elu"))
    return nn.Sequential(*[m.double() for m in layers])
