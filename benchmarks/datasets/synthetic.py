"""Synthetic monotone-regression targets with a complexity knob (depth probe).

Every target is non-decreasing in every input, asserted numerically at
generation time. Inputs are sampled from the **centered** cube
``[-1,1]^d`` so that sharp/smooth nonlinearities actually bite (a positive
cube ``[0,1]^d`` keeps every non-negative-weight preactivation positive,
collapsing ReLU/ELU to the identity — see the module history and
``.superpowers/sdd/task-3-report.md``).

Four families, all monotone by construction (non-negative weights + a
monotone activation, or nested ``min``/``max`` of monotone terms) and
genuinely nonlinear at high complexity ``c`` (each high-``c`` target has
ordinary-least-squares ``R^2 < 0.7`` against its own inputs — the
non-degeneracy gate enforced in the test suite):

* ``additive`` — a per-feature sum of non-negative-weighted ReLU ramps.
  ``c`` concentrates the ramp knots toward the top of the input range and
  removes the linear base term, turning each 1-D contribution from a gentle
  slope (low ``c``, nearly linear) into a flat-then-steep hinge (high ``c``,
  strongly nonlinear).
* ``teacher_relu`` / ``teacher_elu`` — a seeded monotone equal-width
  feed-forward net of depth ``c`` with non-negative weights and a
  **strongly negative bias**, so a large fraction of each layer's
  preactivations are clipped; depth then compounds that clipping into a
  sharply nonlinear function. ``teacher_relu`` (hard clip at 0) and
  ``teacher_elu`` (smooth floor at ``-1``) produce genuinely different
  targets on the centered domain.
* ``lattice`` — an alternating ``max``/``min`` nesting (a tropical /
  min-max lattice) over ``2^c`` monotone single-hidden-layer ReLU experts.
  ``c`` controls the nesting depth and the number of experts; ``min``/``max``
  of monotone functions is monotone, and nesting compounds the piecewise
  interaction structure without oscillation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt

from benchmarks._common.bundle import DatasetBundle

if TYPE_CHECKING:
    from collections.abc import Callable

_Kind = Literal["additive", "teacher_relu", "teacher_elu", "lattice"]
Array = npt.NDArray[np.floating]

_TEACHER_WIDTH = 6
_LATTICE_WIDTH = 4


def _relu(z: Array) -> Array:
    """Elementwise ReLU, a sharp piecewise-linear monotone-increasing activation."""
    return np.maximum(0.0, z)


def _elu(z: Array) -> Array:
    """Elementwise ELU (alpha=1), a smooth monotone-increasing activation."""
    # np.where evaluates both branches eagerly, so clip the expm1 argument to
    # <= 0 to avoid a spurious overflow warning on the (discarded) branch
    # where z is large and positive.
    return np.where(z > 0, z, np.expm1(np.minimum(z, 0.0)))


def _sample_inputs(rng: np.random.Generator, n: int, d: int) -> Array:
    """Draw ``n`` inputs from the centered cube ``[-1,1]^d``.

    Centering (rather than ``[0,1]^d``) is what lets ReLU/ELU preactivations
    straddle zero, so the nonlinearity is exercised; monotonicity is
    unaffected (non-negative weights keep every family non-decreasing on any
    input range).

    :param rng: Seeded generator.
    :param n: Row count.
    :param d: Input dimension.
    :returns: ``(n, d)`` array in ``[-1,1]``.
    """
    return rng.uniform(-1.0, 1.0, size=(n, d))


def _additive(d: int, c: int, seed: int) -> Callable[[Array], Array]:
    """Build a per-feature sum of non-negative-weighted ReLU ramps.

    Each feature's contribution is ``g_j(x) = b0_j * x + sum_k w_jk *
    relu(x - knot_jk)`` with ``b0_j, w_jk >= 0``, a non-negative-weighted sum
    of non-decreasing pieces, hence monotone non-decreasing in ``x``.

    The complexity knob ``c`` shapes the ramps: the knot lower bound rises
    with ``c`` (``knot_lo = -1.5 + 0.5*c``, so knots concentrate near the top
    of the ``[-1,1]`` input range) and the linear base slope shrinks to zero
    (``base_hi = max(0, 0.6 - 0.15*c)``). Low ``c`` is a gently-sloped,
    nearly-linear function; high ``c`` is a flat-then-steep hinge stack that
    ordinary least squares cannot fit (``R^2 < 0.7``).

    :param d: Input dimension.
    :param c: Complexity knob.
    :param seed: RNG seed.
    :returns: The target function.
    """
    rng = np.random.default_rng(seed)
    n_knots = 5
    knot_lo = -1.5 + 0.5 * c
    base_hi = max(0.0, 0.6 - 0.15 * c)
    knots = rng.uniform(knot_lo, 1.0, size=(d, n_knots))
    base_slope = rng.uniform(0.0, base_hi, size=d)
    ramp_weights = rng.uniform(0.3, 1.0, size=(d, n_knots))

    def f(x: Array) -> Array:
        out = np.zeros(len(x))
        for j in range(d):
            xj = x[:, j]
            out += base_slope[j] * xj
            for k in range(n_knots):
                out += ramp_weights[j, k] * np.maximum(0.0, xj - knots[j, k])
        return out

    return f


def _teacher(
    d: int, depth: int, seed: int, act: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """Build a seeded monotone equal-width feed-forward net of depth ``depth``.

    Input projection ``h = X @ W_in + b_in`` (``W_in >= 0``) lifts to width
    :data:`_TEACHER_WIDTH`. Each of ``depth`` hidden layers computes ``h =
    act(h @ M_l + b_l)`` with ``M_l >= 0`` and a **strongly negative** bias
    ``b_l in [-3, -0.5]``, so a large fraction of preactivations are clipped
    by ``act``; stacking depth compounds that clipping into a sharply
    nonlinear map. Output ``y = h @ W_out`` (``W_out >= 0``).

    Monotone because ``W_in, M_l, W_out`` are non-negative and ``act`` is
    monotone-increasing, so every layer is non-decreasing in ``X`` on any
    input range. There is no residual skip: on the centered domain the deep
    non-negative stack does not collapse (verified by the non-degeneracy
    test), and dropping the skip is what lets ``act`` shape the output.

    :param d: Input dimension.
    :param depth: Number of hidden layers (the complexity knob ``c``).
    :param seed: RNG seed.
    :param act: Monotone-increasing activation (:func:`_relu` or
        :func:`_elu`).
    :returns: The target function.
    """
    rng = np.random.default_rng(seed)
    width = _TEACHER_WIDTH
    w_in = rng.uniform(0.0, 2.0, size=(d, width))
    b_in = rng.uniform(-1.0, 1.0, size=width)
    ms = [rng.uniform(0.0, 1.0, size=(width, width)) for _ in range(depth)]
    bs = [rng.uniform(-3.0, -0.5, size=width) for _ in range(depth)]
    w_out = rng.uniform(0.0, 1.0, size=(width, 1))

    def f(x: Array) -> Array:
        h: Array = x @ w_in + b_in
        for m, bnd in zip(ms, bs, strict=True):
            h = act(h @ m + bnd)
        y = h @ w_out
        return np.asarray(y[:, 0])

    return f


def _lattice(d: int, depth: int, seed: int) -> Callable[[Array], Array]:
    """Build an alternating ``max``/``min`` lattice over monotone ReLU experts.

    Each of ``2^depth`` experts is a monotone single-hidden-layer ReLU net
    (``expert(x) = relu((X @ W_in + b_in) @ M + b) @ w_out`` with all weight
    matrices non-negative and a strongly negative hidden bias, so the expert
    is sharply nonlinear). The experts are combined by an alternating
    pairwise reduction — ``max`` at even levels, ``min`` at odd levels — until
    one value remains. ``min``/``max`` of monotone functions is monotone, so
    the whole lattice is monotone, and the alternating nesting builds a
    non-convex piecewise interaction that ordinary least squares cannot fit
    at high ``depth``.

    :param d: Input dimension.
    :param depth: Nesting depth / expert-count exponent (the complexity knob
        ``c``); the lattice uses ``2^max(1, depth)`` experts.
    :param seed: RNG seed.
    :returns: The target function.
    """
    rng = np.random.default_rng(seed)
    width = _LATTICE_WIDTH
    n_experts = 2 ** max(1, depth)
    experts = []
    for _ in range(n_experts):
        w_in = rng.uniform(0.0, 2.0, size=(d, width))
        b_in = rng.uniform(-1.0, 1.0, size=width)
        m = rng.uniform(0.0, 1.0, size=(width, width))
        b = rng.uniform(-3.0, -0.5, size=width)
        w_out = rng.uniform(0.0, 1.0, size=width)
        experts.append((w_in, b_in, m, b, w_out))

    def _expert(x: Array, e: tuple[Array, Array, Array, Array, Array]) -> Array:
        w_in, b_in, m, b, w_out = e
        h = np.maximum(0.0, (x @ w_in + b_in) @ m + b)
        return np.asarray(h @ w_out)

    def f(x: Array) -> Array:
        cols: Array = np.stack([_expert(x, e) for e in experts], axis=1)
        lvl = 0
        while cols.shape[1] > 1:
            half = cols.shape[1] // 2
            a, b = cols[:, :half], cols[:, half : 2 * half]
            cols = np.maximum(a, b) if lvl % 2 == 0 else np.minimum(a, b)
            lvl += 1
        return np.asarray(cols[:, 0])

    return f


def _target_fn(kind: _Kind, *, c: int, d: int, seed: int) -> Callable[[Array], Array]:
    """Dispatch to the target-family builder named by ``kind``."""
    if kind == "additive":
        return _additive(d, c, seed)
    if kind == "teacher_relu":
        return _teacher(d, c, seed, _relu)
    if kind == "teacher_elu":
        return _teacher(d, c, seed, _elu)
    if kind == "lattice":
        return _lattice(d, c, seed)
    raise ValueError(f"unknown kind {kind!r}")


def _assert_monotone(f: Callable[[Array], Array], d: int, seed: int) -> None:
    """Numerically assert ``f`` is non-decreasing in each input dimension.

    :raises AssertionError: if bumping any single feature up ever lowers
        ``f`` (beyond a small floating-point tolerance).
    """
    rng = np.random.default_rng(seed + 999)
    x = _sample_inputs(rng, 256, d)
    base = f(x)
    for j in range(d):
        xp = x.copy()
        xp[:, j] = x[:, j] + 0.05
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
    :param c: Complexity knob (teacher depth / lattice nesting / additive
        knot concentration).
    :param d: Input dimension.
    :param n_train: Train rows.
    :param n_test: Test rows.
    :param seed: RNG seed.
    :returns: Regression bundle, all features monotone-increasing, inputs in
        ``[-1,1]^d``.
    """
    f = _target_fn(kind, c=c, d=d, seed=seed)
    _assert_monotone(f, d, seed)
    rng = np.random.default_rng(seed)
    x_tr = _sample_inputs(rng, n_train, d)
    x_te = _sample_inputs(rng, n_test, d)
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
