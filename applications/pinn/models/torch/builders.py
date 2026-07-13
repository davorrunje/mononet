# SPDX-License-Identifier: Apache-2.0
"""PyTorch model builders for the four PINN methods.

Every model is a callable ``u(x, t)`` on column vectors. The monotone-in-``x`` /
free-in-``t`` structure is achieved by embedding ``t`` through an *unconstrained*
MLP and feeding ``[x, h(t)]`` to a stack that is monotone-increasing in all
inputs: ``x`` gets its sign via the mask, while ``monotone(arbitrary(t))`` is
arbitrary in ``t``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mononet.core.types import ActivationSpec, MonotonicityMask
from mononet.torch import MonoInput, MonoLinear, MonoResidual

if TYPE_CHECKING:
    from applications.pinn.models.protocol import Method, ModelConfig

Bounds = tuple[float, float, float, float]


def _bounds(domain: tuple[tuple[float, float], tuple[float, float]]) -> Bounds:
    """Flatten a ``((x0,x1),(t0,t1))`` domain into ``(x0,x1,t0,t1)`` floats."""
    (x0, x1), (t0, t1) = domain
    return (float(x0), float(x1), float(t0), float(t1))


def _normalize(
    x: torch.Tensor, t: torch.Tensor, bounds: Bounds
) -> tuple[torch.Tensor, torch.Tensor]:
    """Affine-map ``x``, ``t`` from the problem domain to ``[-1, 1]``.

    Done inside the model (so it is a function of raw ``(x, t)`` and the PINN
    residual's autodiff chains through it); increasing in ``x``, so monotonicity
    direction is preserved. Mirrors the JAX builder.
    """
    x0, x1, t0, t1 = bounds
    return 2.0 * (x - x0) / (x1 - x0) - 1.0, 2.0 * (t - t0) / (t1 - t0) - 1.0


def _plain_mlp(in_dim: int, width: int, out_dim: int, activation: str) -> nn.Module:
    act = {"tanh": nn.Tanh, "elu": nn.ELU, "softplus": nn.Softplus}[activation]
    return nn.Sequential(
        nn.Linear(in_dim, width),
        act(),
        nn.Linear(width, width),
        act(),
        nn.Linear(width, out_dim),
    )


class _ClampedLinear(nn.Module):
    """Linear layer with non-negative weights (inexpressive monotone map)."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``x @ |W|^T + b`` (weights forced non-negative)."""
        out: torch.Tensor = torch.nn.functional.linear(x, self.weight.abs(), self.bias)
        return out


class VanillaMLP(nn.Module):
    """Unconstrained MLP over ``[x, t]`` (also the ``soft`` architecture)."""

    def __init__(self, cfg: ModelConfig, bounds: Bounds) -> None:
        """Build the unconstrained MLP."""
        super().__init__()
        self.bounds = bounds
        self.net = _plain_mlp(2, cfg.width, 1, cfg.plain_activation)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``u(x, t)``."""
        x, t = _normalize(x, t, self.bounds)
        out: torch.Tensor = self.net(torch.cat([x, t], dim=-1))
        return out


class WeightClipMono(nn.Module):
    """Monotone-in-``x`` net via non-negative weights (inexpressive baseline)."""

    def __init__(self, cfg: ModelConfig, sign_x: int, bounds: Bounds) -> None:
        """Build the clamped-weight monotone stack and the ``t`` embedding."""
        super().__init__()
        self.bounds = bounds
        self.sign_x = float(sign_x)
        self.t_embed = _plain_mlp(
            1, cfg.t_embed_width, cfg.t_embed_dim, cfg.plain_activation
        )
        in_dim = 1 + cfg.t_embed_dim
        self.stack = nn.Sequential(
            _ClampedLinear(in_dim, cfg.width),
            nn.Softplus(),
            _ClampedLinear(cfg.width, cfg.width),
            nn.Softplus(),
            _ClampedLinear(cfg.width, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``u(x, t)`` (monotone in ``x`` via non-negative weights)."""
        x, t = _normalize(x, t, self.bounds)
        z = torch.cat([self.sign_x * x, self.t_embed(t)], dim=-1)
        out: torch.Tensor = self.stack(z)
        return out


class HardMonoField(nn.Module):
    """Expressive hard-monotone-in-``x`` field built from ``mononet`` layers."""

    def __init__(self, cfg: ModelConfig, sign_x: int, bounds: Bounds) -> None:
        """Build the mononet monotone stack and the unconstrained ``t`` embedding."""
        super().__init__()
        self.bounds = bounds
        self.t_embed = _plain_mlp(
            1, cfg.t_embed_width, cfg.t_embed_dim, cfg.plain_activation
        )
        in_dim = 1 + cfg.t_embed_dim
        mask = MonotonicityMask(
            np.array([sign_x, *([1] * cfg.t_embed_dim)], dtype=np.int8)
        )
        self.mono_input = MonoInput(mask)
        act = ActivationSpec(cfg.mono_activation)  # type: ignore[arg-type]
        blocks: list[nn.Module] = [
            MonoResidual(in_dim, cfg.width, mode=cfg.mode, activation=act)  # type: ignore[arg-type]
        ]
        blocks.extend(
            MonoResidual(cfg.width, cfg.width, mode=cfg.mode, activation=act)  # type: ignore[arg-type]
            for _ in range(cfg.n_blocks - 1)
        )
        blocks.append(MonoLinear(cfg.width, 1, mode=cfg.mode, activation="identity"))  # type: ignore[arg-type]
        self.mono = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return ``u(x, t)`` (monotone in ``x`` by construction)."""
        x, t = _normalize(x, t, self.bounds)
        z = torch.cat([x, self.t_embed(t)], dim=-1)
        out: torch.Tensor = self.mono(self.mono_input(z))
        return out


def build_torch(problem: object, cfg: ModelConfig, method: Method) -> nn.Module:
    """Build a torch model for ``method`` given a problem's monotonicity sign.

    :param problem: A registered `Problem` (its admissibility mask sets ``sign_x``).
    :param cfg: Architecture configuration.
    :param method: One of ``vanilla`` / ``soft`` / ``weight_clip`` / ``hard_monotone``.
    :returns: A torch ``nn.Module`` with ``forward(x, t)``.
    :raises ValueError: If ``method`` is unknown.
    """
    torch.manual_seed(cfg.seed)
    sign_x = int(problem.admissibility().mask[0])  # type: ignore[attr-defined]
    bounds = _bounds(problem.domain)  # type: ignore[attr-defined]
    if method in ("vanilla", "soft"):
        return VanillaMLP(cfg, bounds)
    if method == "weight_clip":
        return WeightClipMono(cfg, sign_x, bounds)
    if method == "hard_monotone":
        return HardMonoField(cfg, sign_x, bounds)
    raise ValueError(f"unknown method {method!r}")
