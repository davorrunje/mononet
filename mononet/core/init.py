# SPDX-License-Identifier: Apache-2.0
"""Static, data-free initialization derivation for the ``absolute`` construction.

Pure NumPy (no backend import). Derives, from an activation's moments under a
standard normal pre-activation, a variance-preserving weight ``gain`` and a
layer-mean-centering ``bias`` for ``mode="absolute"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from mononet.core.types import ActivationSpec

_GH_DEG = 64  # Gauss-Hermite nodes for E_{H~N(0,1)}[.]
# probabilists' Gauss-Hermite: E[f(H)] = sum(_W * f(_X)), H ~ N(0, 1)
_X, _W_RAW = np.polynomial.hermite_e.hermegauss(_GH_DEG)
_W = _W_RAW / np.sqrt(2.0 * np.pi)


def _act(name: str, h: np.ndarray) -> np.ndarray:
    """NumPy mirror of the backend base activations.

    :param name: One of ``relu``, ``elu``, ``selu``, ``softplus``.
    :param h: Pre-activation values.
    :returns: Activated values, same shape as ``h``.
    :raises ValueError: If ``name`` is not a known activation.
    """
    if name == "relu":
        return np.asarray(np.maximum(0.0, h))
    if name == "elu":
        return np.where(h > 0.0, h, np.expm1(h))
    if name == "selu":
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(h > 0.0, h, alpha * np.expm1(h))
    if name == "softplus":
        return np.asarray(np.logaddexp(0.0, h))
    raise ValueError(f"unknown activation {name!r}")


def _expect(name: str, mean: float, scale: float, *, moment: int) -> float:
    """``E_{H~N(0,1)}[ act(scale*H + mean)^moment ]`` via Gauss-Hermite.

    :param name: Activation name.
    :param mean: Added to the scaled node (the pre-activation mean).
    :param scale: Multiplies the node (the pre-activation std).
    :param moment: 1 for the mean, 2 for the second moment.
    :returns: The expectation as a float.
    """
    vals = _act(name, scale * _X + mean)
    return float(np.sum(_W * vals**moment))


def _variance(name: str, mean: float, scale: float) -> float:
    return (
        _expect(name, mean, scale, moment=2) - _expect(name, mean, scale, moment=1) ** 2
    )


def _bisect(
    f: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    tol: float = 1e-9,
    iters: int = 200,
) -> float:
    """Bisection root of a monotone ``f`` on ``[lo, hi]`` (``f(lo)``/``f(hi)`` bracket 0)."""
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if (fmid > 0.0) == (flo > 0.0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _solve_gain(name: str, bias: float) -> float:
    """Gain s.t. ``Var[act(gain*H + bias)] = 1`` (variance is increasing in gain)."""
    return _bisect(lambda g: _variance(name, bias, g) - 1.0, 1e-4, 20.0)


def _solve_bias(name: str, gain: float, f: float) -> float:
    """Scalar bias s.t. the layer mean ``f*E[act(H+b)] - (1-f)*E[act(-(H+b))] = 0``.

    Monotone increasing in ``b`` (both terms shift the layer mean up with ``b``).
    """

    def layer_mean(b: float) -> float:
        conv = _expect(name, b, gain, moment=1)
        conc = _expect(name, -b, gain, moment=1)  # E[act(-H - b)] via H symmetry
        return f * conv - (1.0 - f) * conc

    return _bisect(layer_mean, -20.0, 20.0)


def absolute_init_params(
    activation: ActivationSpec | str, convex_fraction: float
) -> tuple[float, float]:
    """Derive ``(gain, bias)`` for the ``absolute`` construction.

    The weight init std is ``gain / sqrt(fan_in)`` (variance-preserving through the
    ``|W|`` + convex/concave-activation map), and the whole bias vector is
    initialised to the scalar ``bias`` (which centers a layer's output mean). At
    ``convex_fraction == 0.5`` the split is self-cancelling so ``bias == 0`` and the
    default fix is purely the gain. Both are data-free (Gauss-Hermite quadrature),
    so the init stays static and seed-reproducible.

    :param activation: Base activation name or :class:`ActivationSpec`.
    :param convex_fraction: Fraction of convex units in the layer.
    :returns: ``(gain, bias)`` — ``gain > 0``; ``bias == 0.0`` when ``convex_fraction == 0.5``.
    :raises ValueError: If the activation is unknown.
    """
    name = activation if isinstance(activation, str) else activation.name
    _act(name, np.zeros(1))  # validate name early (raises on unknown)
    if convex_fraction == 0.5:
        return _solve_gain(name, 0.0), 0.0
    gain = _solve_gain(name, 0.0)
    bias = 0.0
    for _ in range(8):  # fixed-point: gain and bias couple mildly off f=0.5
        bias = _solve_bias(name, gain, convex_fraction)
        gain = _solve_gain(name, bias)
    return gain, bias
