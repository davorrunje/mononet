# SPDX-License-Identifier: Apache-2.0
"""Tests for deterministic sampling and the observation sampler."""

from __future__ import annotations

import numpy as np
import pytest

from applications.pinn.core import sampling

DOMAIN = ((-1.0, 2.0), (0.0, 1.5))


@pytest.mark.parametrize("strategy", ["uniform", "lhs"])
def test_collocation_deterministic_and_in_domain(strategy: str) -> None:
    """Same seed -> identical points; all points lie in the domain."""
    a = sampling.collocation(DOMAIN, 256, seed=0, strategy=strategy)
    b = sampling.collocation(DOMAIN, 256, seed=0, strategy=strategy)
    c = sampling.collocation(DOMAIN, 256, seed=1, strategy=strategy)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (256, 2)
    (x_lo, x_hi), (t_lo, t_hi) = DOMAIN
    assert np.all((a[:, 0] >= x_lo) & (a[:, 0] <= x_hi))
    assert np.all((a[:, 1] >= t_lo) & (a[:, 1] <= t_hi))


def test_unknown_strategy_raises() -> None:
    """An unknown sampling strategy is rejected."""
    with pytest.raises(ValueError, match="unknown strategy"):
        sampling.collocation(DOMAIN, 8, seed=0, strategy="sobol")


def test_initial_points_on_initial_line() -> None:
    """Initial points all have t == t_min."""
    pts = sampling.initial_points(DOMAIN, 64, seed=3)
    assert pts.shape == (64, 2)
    assert np.all(pts[:, 1] == DOMAIN[1][0])


def test_boundary_points_on_boundaries() -> None:
    """Boundary points lie on x_min or x_max, split evenly."""
    pts = sampling.boundary_points(DOMAIN, 40, seed=2)
    assert pts.shape == (40, 2)
    on_edge = np.isclose(pts[:, 0], DOMAIN[0][0]) | np.isclose(pts[:, 0], DOMAIN[0][1])
    assert np.all(on_edge)
    assert np.isclose(pts[:, 0], DOMAIN[0][0]).sum() == 20


def test_eval_grid_axes() -> None:
    """Eval grid axes span the domain with the requested counts."""
    x_values, t_values = sampling.eval_grid(DOMAIN, 50, 30)
    assert x_values.shape == (50,)
    assert t_values.shape == (30,)
    assert x_values[0] == DOMAIN[0][0]
    assert t_values[-1] == DOMAIN[1][1]


def test_observations_noiseless_match_field() -> None:
    """With zero noise, observations equal the sampled field values."""
    x_values, t_values = sampling.eval_grid(DOMAIN, 20, 10)
    field = np.outer(t_values, x_values)  # arbitrary reference field
    coords, values = sampling.observations(
        field, x_values, t_values, n_obs=15, noise_std=0.0, seed=7
    )
    assert coords.shape == (15, 2)
    assert values.shape == (15,)
    # each observed value must appear in the field grid
    for (x, t), v in zip(coords, values, strict=True):
        i = int(np.argmin(np.abs(t_values - t)))
        j = int(np.argmin(np.abs(x_values - x)))
        assert np.isclose(v, field[i, j])


def test_observations_deterministic_and_noise_reproducible() -> None:
    """Same seed reproduces coordinates and noisy values exactly."""
    x_values, t_values = sampling.eval_grid(DOMAIN, 20, 10)
    field = np.outer(t_values, x_values)
    c1, v1 = sampling.observations(
        field, x_values, t_values, n_obs=15, noise_std=0.1, seed=7
    )
    c2, v2 = sampling.observations(
        field, x_values, t_values, n_obs=15, noise_std=0.1, seed=7
    )
    assert np.array_equal(c1, c2)
    assert np.array_equal(v1, v2)
