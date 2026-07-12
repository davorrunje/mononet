# SPDX-License-Identifier: Apache-2.0
"""Deterministic point sampling for PINN training and evaluation.

All point sets are generated once in NumPy from a seed so JAX and PyTorch train
on identical inputs. Provides interior collocation, initial/boundary points, a
structured evaluation grid, and — for the inverse flagship — a sparse, noisy
observation sampler drawn from a reference field.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from applications._common import seeding

Array = npt.NDArray[np.floating]
Domain = tuple[tuple[float, float], tuple[float, float]]


def _uniform(gen: np.random.Generator, lo: float, hi: float, n: int) -> Array:
    return lo + (hi - lo) * gen.random(n)


def _latin(gen: np.random.Generator, lo: float, hi: float, n: int) -> Array:
    """One-dimensional Latin-hypercube sample of ``n`` points in ``[lo, hi]``."""
    edges = np.arange(n)
    jitter = gen.random(n)
    unit = (edges + jitter) / n
    return lo + (hi - lo) * gen.permutation(unit)


def collocation(
    domain: Domain, n: int, *, seed: int, strategy: str = "uniform"
) -> Array:
    """Sample ``n`` interior collocation points ``(x, t)``.

    :param domain: ``((x_min, x_max), (t_min, t_max))``.
    :param n: Number of points.
    :param seed: RNG seed.
    :param strategy: ``"uniform"`` or ``"lhs"`` (Latin hypercube).
    :returns: Array of shape ``(n, 2)`` with columns ``[x, t]``.
    :raises ValueError: If ``strategy`` is unknown.
    """
    gen = seeding.rng(seed)
    (x_lo, x_hi), (t_lo, t_hi) = domain
    if strategy == "uniform":
        x = _uniform(gen, x_lo, x_hi, n)
        t = _uniform(gen, t_lo, t_hi, n)
    elif strategy == "lhs":
        x = _latin(gen, x_lo, x_hi, n)
        t = _latin(gen, t_lo, t_hi, n)
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    return np.column_stack([x, t])


def initial_points(domain: Domain, n: int, *, seed: int) -> Array:
    """Sample ``n`` points on the initial line ``t = t_min``.

    :param domain: ``((x_min, x_max), (t_min, t_max))``.
    :param n: Number of points.
    :param seed: RNG seed.
    :returns: Array of shape ``(n, 2)`` with ``t == t_min``.
    """
    gen = seeding.rng(seed)
    (x_lo, x_hi), (t_lo, _t_hi) = domain
    x = _uniform(gen, x_lo, x_hi, n)
    return np.column_stack([x, np.full(n, t_lo)])


def boundary_points(domain: Domain, n: int, *, seed: int) -> Array:
    """Sample ``n`` points split across the two spatial boundaries.

    :param domain: ``((x_min, x_max), (t_min, t_max))``.
    :param n: Total number of points (half per boundary).
    :param seed: RNG seed.
    :returns: Array of shape ``(n, 2)`` with ``x`` at ``x_min`` or ``x_max``.
    """
    gen = seeding.rng(seed)
    (x_lo, x_hi), (t_lo, t_hi) = domain
    n_left = n // 2
    t_left = _uniform(gen, t_lo, t_hi, n_left)
    t_right = _uniform(gen, t_lo, t_hi, n - n_left)
    left = np.column_stack([np.full(n_left, x_lo), t_left])
    right = np.column_stack([np.full(n - n_left, x_hi), t_right])
    return np.vstack([left, right])


def eval_grid(domain: Domain, nx: int, nt: int) -> tuple[Array, Array]:
    """Return the structured evaluation grid axes ``(x_values, t_values)``.

    :param domain: ``((x_min, x_max), (t_min, t_max))``.
    :param nx: Number of spatial samples.
    :param nt: Number of temporal samples.
    :returns: ``(x_values, t_values)`` of shapes ``(nx,)`` and ``(nt,)``.
    """
    (x_lo, x_hi), (t_lo, t_hi) = domain
    return np.linspace(x_lo, x_hi, nx), np.linspace(t_lo, t_hi, nt)


def observations(
    field: Array,
    x_values: Array,
    t_values: Array,
    *,
    n_obs: int,
    noise_std: float,
    seed: int,
) -> tuple[Array, Array]:
    """Draw sparse, noisy observations from a reference field.

    Emulates scattered probe / loop-detector data: ``n_obs`` grid cells are
    sampled without replacement and their values perturbed by Gaussian noise.

    :param field: Reference field of shape ``(len(t_values), len(x_values))``.
    :param x_values: Spatial grid axis.
    :param t_values: Temporal grid axis.
    :param n_obs: Number of observations.
    :param noise_std: Standard deviation of additive Gaussian noise.
    :param seed: RNG seed.
    :returns: ``(coords, values)`` with ``coords`` of shape ``(n_obs, 2)``
        (columns ``[x, t]``) and ``values`` of shape ``(n_obs,)``.
    """
    gen = seeding.rng(seed)
    nt, nx = field.shape
    flat = gen.choice(nt * nx, size=n_obs, replace=False)
    ti, xi = np.divmod(flat, nx)
    coords = np.column_stack([x_values[xi], t_values[ti]])
    values = field[ti, xi] + noise_std * gen.standard_normal(n_obs)
    return coords, values
