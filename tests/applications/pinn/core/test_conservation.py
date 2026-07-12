# SPDX-License-Identifier: Apache-2.0
"""Tests for the conservation-law problems and their registration."""

from __future__ import annotations

import numpy as np

from applications.pinn.core import problems
from applications.pinn.core.problems import conservation


def test_all_conservation_problems_registered() -> None:
    """The four forward-tier problems are discoverable via the registry."""
    for key in ("burgers_riemann", "burgers_smooth", "advection", "lwr_riemann"):
        assert key in problems.available()
        assert problems.get(key).key == key


def test_burgers_riemann_mask_matches_monotone_direction() -> None:
    """A decreasing Riemann (u_l>u_r) is non-increasing in x; increasing flips."""
    dec = conservation.BurgersRiemann(u_l=1.0, u_r=0.0)
    inc = conservation.BurgersRiemann(u_l=0.0, u_r=1.0)
    assert dec.admissibility().mask == (-1, 0)
    assert inc.admissibility().mask == (1, 0)


def test_lwr_mask_matches_density_direction() -> None:
    """Forming queue (rho_l<rho_r) is non-decreasing in x."""
    assert conservation.LwrRiemann(rho_l=0.2, rho_r=0.8).admissibility().mask == (1, 0)
    assert conservation.LwrRiemann(rho_l=0.8, rho_r=0.2).admissibility().mask == (-1, 0)


def test_flux_values() -> None:
    """Flux and characteristic speed match the analytic forms."""
    b = conservation.BurgersRiemann()
    u = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(b.flux(u), 0.5 * u**2)
    assert np.allclose(b.flux_prime(u), u)
    adv = conservation.LinearAdvection(a=0.7)
    assert np.allclose(adv.flux_prime(u), 0.7)


def test_advection_residual_is_small() -> None:
    """The exact advection solution satisfies u_t + a u_x = 0 (finite diff)."""
    adv = conservation.LinearAdvection(a=0.8)
    x = np.linspace(-2.0, 3.0, 400)
    t, dt, dx = 0.5, 1e-4, x[1] - x[0]
    t_plus = np.full_like(x, t + dt)
    t_minus = np.full_like(x, t - dt)
    u_t = (adv.ground_truth(x, t_plus) - adv.ground_truth(x, t_minus)) / (2 * dt)
    f = adv.flux(adv.ground_truth(x, np.full_like(x, t)))
    f_x = np.gradient(f, dx)
    interior = slice(5, -5)
    assert np.max(np.abs((u_t + f_x)[interior])) < 1e-2


def test_burgers_smooth_ground_truth_matches_characteristic_pre_breaking() -> None:
    """Before the breaking time, the Godunov ground truth ~ the characteristic."""
    p = conservation.BurgersSmoothShock(steepness=1.0)  # t_b = 1.0
    x = np.linspace(-4.0, 4.0, 300)
    t = 0.5
    gt = p.ground_truth(x, np.full_like(x, t))
    char = exact_characteristic(p, x, t)
    assert np.max(np.abs(gt - char)) < 2e-2
    # and it is monotone non-increasing in x
    assert np.all(np.diff(gt) <= 1e-6)


def exact_characteristic(
    p: conservation.BurgersSmoothShock, x: np.ndarray, t: float
) -> np.ndarray:
    """Return the pre-breaking characteristic solution for the smooth problem."""
    from applications.pinn.core import exact

    return exact.burgers_characteristic(x, t, p._u0)


def test_burgers_smooth_develops_shock_and_stays_monotone() -> None:
    """Past the breaking time the profile is a monotone (non-oscillating) shock."""
    p = conservation.BurgersSmoothShock(steepness=1.0)
    x = np.linspace(-4.0, 4.0, 400)
    gt = p.ground_truth(x, np.full_like(x, 1.8))  # t > t_b = 1.0
    assert np.all(np.diff(gt) <= 1e-6)  # monotone, no overshoot
