# SPDX-License-Identifier: Apache-2.0
"""Cross-backend parity for the PINN models.

Exact *bitwise* cross-backend equivalence of the underlying layers is mononet's
responsibility and is covered by ``tests/equivalence/``. Here we assert the
*application-level* parity that matters: with identical configuration and inputs,
both backends' hard-monotone field satisfies the same by-construction constraint
(monotone in ``x``) and produces a finite field of the same shape. Numeric
agreement of *trained* runs across backends is an empirical result reported in
the experiment sweep, not a unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
import torch

from applications.pinn.core.admissibility import violation
from applications.pinn.core.problems import conservation
from applications.pinn.models.jax import builders as jax_builders
from applications.pinn.models.protocol import ModelConfig
from applications.pinn.models.torch import builders as torch_builders

CFG = ModelConfig(width=16, n_blocks=2, t_embed_dim=4, seed=0)


def test_both_backends_hard_monotone_parity() -> None:
    """Both backends' hard-monotone field is monotone in x on identical inputs."""
    problem = conservation.BurgersRiemann(u_l=1.0, u_r=0.0)  # sign_x = -1
    x_np = np.linspace(-1.5, 2.0, 80, dtype=np.float32)
    t_val = 0.4

    torch_model = torch_builders.build_torch(problem, CFG, "hard_monotone")
    xt = torch.as_tensor(x_np).reshape(-1, 1)
    tt = torch.full_like(xt, t_val)
    u_torch = torch_model(xt, tt).detach().numpy().ravel()

    jax_model = jax_builders.build_jax(problem, CFG, "hard_monotone")
    xj = jnp.asarray(x_np).reshape(-1, 1)
    tj = jnp.full_like(xj, t_val)
    u_jax = np.asarray(jax_model(xj, tj)).ravel()

    assert u_torch.shape == u_jax.shape == (80,)
    assert np.all(np.isfinite(u_torch))
    assert np.all(np.isfinite(u_jax))
    # the shared by-construction guarantee holds in both backends
    assert violation(u_torch, axis=0, sign=-1) < 1e-4
    assert violation(u_jax, axis=0, sign=-1) < 1e-4


def test_both_backends_respect_increasing_sign() -> None:
    """A forming-queue LWR problem is non-decreasing in x in both backends."""
    problem = conservation.LwrRiemann(rho_l=0.2, rho_r=0.8)  # sign_x = +1
    x_np = np.linspace(-1.5, 1.5, 60, dtype=np.float32)

    ut = (
        torch_builders.build_torch(problem, CFG, "hard_monotone")(
            torch.as_tensor(x_np).reshape(-1, 1),
            torch.full((60, 1), 0.3),
        )
        .detach()
        .numpy()
        .ravel()
    )
    uj = np.asarray(
        jax_builders.build_jax(problem, CFG, "hard_monotone")(
            jnp.asarray(x_np).reshape(-1, 1), jnp.full((60, 1), 0.3)
        )
    ).ravel()

    assert violation(ut, axis=0, sign=1) < 1e-4
    assert violation(uj, axis=0, sign=1) < 1e-4
