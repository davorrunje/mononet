# SPDX-License-Identifier: Apache-2.0
"""Smoke test for the single-run experiment orchestrator."""

from __future__ import annotations

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")

import numpy as np

from applications.pinn.experiments.config import RunConfig
from applications.pinn.experiments.run import run_one
from applications.pinn.models.protocol import ModelConfig

_TINY = ModelConfig(width=8, n_blocks=2, t_embed_dim=4, seed=0)


def _cfg(method: str) -> RunConfig:
    return RunConfig(
        problem="burgers_riemann",
        method=method,  # type: ignore[arg-type]
        backend="jax",
        seed=0,
        model=_TINY,
        n_collocation=128,
        n_ic=32,
        n_bc=16,
        steps=15,
        eval_nx=40,
        eval_nt=3,
    )


def test_run_one_produces_valid_artifact() -> None:
    """A tiny run returns an artifact with finite headline metrics."""
    art = run_one(_cfg("hard_monotone"))
    assert art["problem"] == "burgers_riemann"
    assert art["method"] == "hard_monotone"
    for key in ("l1", "l2", "admissibility_violation", "overshoot"):
        assert np.isfinite(art[key])
    # hard-monotone is admissible by construction
    assert art["admissibility_violation"] < 1e-4


def test_soft_baseline_runs() -> None:
    """The soft baseline runs end-to-end (penalty path exercised)."""
    art = run_one(_cfg("soft"))
    assert np.isfinite(art["l2"])
