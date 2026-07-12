"""Synthetic monotone-regression targets with a complexity knob (depth probe).

Every target is non-decreasing in every input on ``[0,1]^d`` (asserted
numerically at generation time). Three families: ``additive`` (control — a
sum of per-feature non-negative-weighted ReLU ramps, monotone by
construction), ``teacher`` (a seeded monotone MLP of depth ``c`` —
non-negative weights + a monotone activation, so the composite is monotone),
``lattice`` (nested element-wise max/min of monotone linear terms, nesting
depth ``c`` — piecewise complexity without oscillation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from benchmarks._common.bundle import DatasetBundle

if TYPE_CHECKING:
    from collections.abc import Callable

_Kind = Literal["additive", "teacher", "lattice"]
Array = npt.NDArray[np.floating]


def _softplus(z: Array) -> Array:
    """Elementwise softplus, a smooth monotone-increasing activation."""
    return np.logaddexp(0.0, z)


def _additive(d: int, seed: int) -> Callable[[Array], Array]:
    """Build a per-feature sum of non-negative-weighted ReLU ramps.

    Each feature's contribution is ``g_j(x) = b0_j * x + sum_k w_jk *
    relu(x - knot_jk)`` with ``b0_j, w_jk >= 0``, a non-negative-weighted sum
    of non-decreasing pieces, hence monotone non-decreasing in ``x`` by
    construction (independent of knot placement or per-piece weights).
    """
    rng = np.random.default_rng(seed)
    knots = rng.uniform(0, 1, size=(d, 5))
    base_slope = rng.uniform(0.2, 1.0, size=d)
    ramp_weights = rng.uniform(0.0, 1.0, size=(d, 5))

    def f(x: Array) -> Array:
        out = np.zeros(len(x))
        for j in range(d):
            xj = x[:, j]
            out += base_slope[j] * xj
            for k in range(knots.shape[1]):
                out += ramp_weights[j, k] * np.maximum(0.0, xj - knots[j, k])
        return out

    return f


def _teacher(d: int, depth: int, seed: int) -> Callable[[Array], Array]:
    """Build a seeded monotone MLP of depth ``depth``.

    Non-negative weights ensure each layer is monotone in its input, and
    softplus is a monotone-increasing activation, so the composite is
    monotone non-decreasing in ``x``.
    """
    rng = np.random.default_rng(seed)
    widths = [d] + [8] * max(1, depth) + [1]
    ws = [
        rng.uniform(0.0, 1.0, size=(widths[i], widths[i + 1]))
        for i in range(len(widths) - 1)
    ]
    bs = [rng.uniform(-0.5, 0.5, size=widths[i + 1]) for i in range(len(widths) - 1)]

    def f(x: Array) -> Array:
        h = x
        for i, (w, bnd) in enumerate(zip(ws, bs, strict=True)):
            h = h @ w + bnd
            if i < len(ws) - 1:
                h = _softplus(h)
        return h[:, 0]

    return f


def _lattice(d: int, depth: int, seed: int) -> Callable[[Array], Array]:
    """Build a nested element-wise max/min lattice of monotone linear terms.

    Each "expert" ``h = x @ w + b`` with non-negative ``w`` is monotone in
    ``x``. Element-wise ``max``/``min`` of monotone functions is itself
    monotone, so nesting ``depth`` levels of max/min preserves monotonicity
    while adding piecewise complexity without oscillation.
    """
    rng = np.random.default_rng(seed)
    m = 2 ** max(1, depth)
    w = rng.uniform(0.0, 1.0, size=(d, m))
    bnd = rng.uniform(-0.5, 0.5, size=m)
    ops = rng.integers(0, 2, size=depth)  # 0=max, 1=min per level

    def f(x: Array) -> Array:
        h = x @ w + bnd  # (n, m), monotone in x
        for lvl in range(depth):
            half = h.shape[1] // 2
            a, b = h[:, :half], h[:, half : 2 * half]
            h = np.maximum(a, b) if ops[lvl] == 0 else np.minimum(a, b)
        return np.asarray(h[:, 0])

    return f


def _target_fn(kind: _Kind, *, c: int, d: int, seed: int) -> Callable[[Array], Array]:
    """Dispatch to the target-family builder named by ``kind``."""
    if kind == "additive":
        return _additive(d, seed)
    if kind == "teacher":
        return _teacher(d, c, seed)
    if kind == "lattice":
        return _lattice(d, c, seed)
    raise ValueError(f"unknown kind {kind!r}")


def _assert_monotone(f: Callable[[Array], Array], d: int, seed: int) -> None:
    """Numerically assert ``f`` is non-decreasing in each input dimension.

    :raises AssertionError: if bumping any single feature up ever lowers
        ``f`` (beyond a small floating-point tolerance).
    """
    rng = np.random.default_rng(seed + 999)
    x = rng.uniform(0, 1, size=(256, d))
    base = f(x)
    for j in range(d):
        xp = x.copy()
        xp[:, j] = np.minimum(1.0, x[:, j] + 0.05)
        if not (f(xp) - base >= -1e-9).all():
            raise AssertionError(f"target not monotone in dim {j}")


def synth_monotone(
    kind: _Kind,
    c: int,
    *,
    d: int = 6,
    n_train: int = 4000,
    n_test: int = 2000,
    seed: int = 0,
) -> DatasetBundle:
    """Build a monotone-regression :class:`DatasetBundle` (see module docstring).

    :param kind: Target family.
    :param c: Complexity knob (teacher/lattice depth; ignored for additive).
    :param d: Input dimension.
    :param n_train: Train rows.
    :param n_test: Test rows.
    :param seed: RNG seed.
    :returns: Regression bundle, all features monotone-increasing.
    """
    f = _target_fn(kind, c=c, d=d, seed=seed)
    _assert_monotone(f, d, seed)
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(0, 1, size=(n_train, d))
    x_te = rng.uniform(0, 1, size=(n_test, d))
    y_tr, y_te = f(x_tr), f(x_te)
    mu, sd = float(y_tr.mean()), float(y_tr.std() or 1.0)
    return DatasetBundle(
        name=f"synth-{kind}-{c}",
        task="regression",
        X_train=x_tr,
        y_train=(y_tr - mu) / sd,
        X_test=x_te,
        y_test=(y_te - mu) / sd,
        mono_increasing=tuple(range(d)),
        mono_decreasing=(),
        feature_names=tuple(f"x{j}" for j in range(d)),
        metadata={"kind": kind, "c": str(c)},
    )
