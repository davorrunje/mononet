# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the Optuna search and the sweep composer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("optuna")

from applications.pinn.experiments import search, sweep
from applications.pinn.experiments.config import RunConfig
from applications.pinn.models.protocol import ModelConfig

if TYPE_CHECKING:
    from pathlib import Path

_TEMPLATE = RunConfig(
    problem="burgers_riemann",
    method="hard_monotone",
    backend="jax",
    model=ModelConfig(width=8, n_blocks=2, t_embed_dim=4),
    n_collocation=96,
    n_ic=24,
    n_bc=12,
    steps=8,
    eval_nx=30,
    eval_nt=3,
)


def test_search_returns_best_params_and_freezes(tmp_path: Path) -> None:
    """A 2-trial search yields best params and freezes a reloadable config."""
    best = search.search(
        "burgers_riemann",
        "hard_monotone",
        "jax",
        n_trials=2,
        template=_TEMPLATE,
        seed=0,
    )
    assert "lr" in best
    assert "width" in best
    path = search.freeze(
        "burgers_riemann", "hard_monotone", "jax", best, configs_dir=tmp_path
    )
    assert path.exists()
    cfg = search.tuned_config("burgers_riemann", "hard_monotone", "jax", path)
    assert cfg.model.width in (16, 32, 64)


def test_soft_search_includes_penalty() -> None:
    """The soft method's search space includes its penalty weight."""
    soft_template = _TEMPLATE.__class__(
        problem="burgers_riemann",
        method="soft",
        backend="jax",
        model=_TEMPLATE.model,
        n_collocation=96,
        n_ic=24,
        n_bc=12,
        steps=8,
        eval_nx=30,
        eval_nt=3,
    )
    best = search.search(
        "burgers_riemann", "soft", "jax", n_trials=2, template=soft_template, seed=0
    )
    assert "soft_penalty" in best


def test_sweep_runs_small_matrix() -> None:
    """The sweep composes run_one over a 2-cell matrix."""
    configs = sweep.build_matrix(
        ["burgers_riemann"],
        ["hard_monotone", "vanilla"],
        ["jax"],
        [0],
        template=_TEMPLATE,
    )
    assert len(configs) == 2
    results = sweep.run_matrix(configs)
    assert len(results) == 2
    assert {r["method"] for r in results} == {"hard_monotone", "vanilla"}
