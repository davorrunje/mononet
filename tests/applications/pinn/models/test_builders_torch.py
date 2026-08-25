# SPDX-License-Identifier: Apache-2.0
"""Tests for the PyTorch model builders (monotonicity by construction)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from applications.pinn.core.problems import conservation
from applications.pinn.models.protocol import Method, ModelConfig
from applications.pinn.models.torch import builders

CFG = ModelConfig(width=16, n_blocks=2, t_embed_dim=4, seed=0)
METHODS: list[Method] = ["vanilla", "soft", "weight_clip", "hard_monotone"]


def _grid(n: int = 80) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.linspace(-1.5, 2.0, n, dtype=torch.float32).reshape(-1, 1)
    t = torch.full((n, 1), 0.3, dtype=torch.float32)
    return x, t


@pytest.mark.parametrize("method", METHODS)
def test_forward_shape(method: Method) -> None:
    """Every method returns a column vector of the right shape."""
    problem = conservation.BurgersRiemann()
    model = builders.build_torch(problem, CFG, method)
    x, t = _grid()
    out = model(x, t)
    assert out.shape == (80, 1)


def test_hard_monotone_decreasing_in_x() -> None:
    """A decreasing problem (u_l>u_r) yields a non-increasing field in x."""
    problem = conservation.BurgersRiemann(u_l=1.0, u_r=0.0)  # sign_x = -1
    model = builders.build_torch(problem, CFG, "hard_monotone")
    x, t = _grid()
    u = model(x, t).detach().numpy().ravel()
    assert np.all(np.diff(u) <= 1e-4)


def test_hard_monotone_increasing_in_x() -> None:
    """A forming-queue LWR problem (rho_l<rho_r) yields a non-decreasing field."""
    problem = conservation.LwrRiemann(rho_l=0.2, rho_r=0.8)  # sign_x = +1
    model = builders.build_torch(problem, CFG, "hard_monotone")
    x, t = _grid()
    u = model(x, t).detach().numpy().ravel()
    assert np.all(np.diff(u) >= -1e-4)


def test_weight_clip_is_monotone_in_x() -> None:
    """The weight-clip baseline is also monotone in x by construction."""
    problem = conservation.BurgersRiemann(u_l=1.0, u_r=0.0)  # sign_x = -1
    model = builders.build_torch(problem, CFG, "weight_clip")
    x, t = _grid()
    u = model(x, t).detach().numpy().ravel()
    assert np.all(np.diff(u) <= 1e-4)


def test_hard_monotone_depends_on_t() -> None:
    """The field genuinely varies with t (free, not constant/separable-trivial)."""
    problem = conservation.BurgersRiemann()
    model = builders.build_torch(problem, CFG, "hard_monotone")
    x = torch.linspace(-1.0, 1.0, 40, dtype=torch.float32).reshape(-1, 1)
    u0 = model(x, torch.zeros_like(x))
    u1 = model(x, torch.ones_like(x))
    assert float((u0 - u1).abs().max().detach()) > 1e-4
