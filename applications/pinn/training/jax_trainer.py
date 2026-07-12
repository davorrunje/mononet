# SPDX-License-Identifier: Apache-2.0
"""JAX (optax) training loop for the PINN methods.

The model is split into a pure ``params`` pytree via ``nnx.split`` so that
``jax.grad`` cleanly provides (a) the input derivatives ``u_x, u_t`` that build
the PDE residual and (b) the parameter gradients of the total loss. First-order
scalar conservation laws need only first derivatives, so no Hessian is taken.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

if TYPE_CHECKING:
    from applications.pinn.training.losses import LossWeights, TrainingData


def train(
    problem: object,
    model: nnx.Module,
    data: TrainingData,
    *,
    weights: LossWeights,
    sign_x: int,
    lr: float = 1e-3,
    steps: int = 200,
) -> tuple[nnx.Module, list[float]]:
    """Train ``model`` with optax Adam on the PINN loss; return it and the history.

    :param problem: Registered problem (supplies ``flux_prime`` for the residual).
    :param model: A Flax NNX model callable as ``u(x, t)``.
    :param data: Collocation and (IC/BC or observation) point sets.
    :param weights: Loss-term weights.
    :param sign_x: Desired monotonicity sign in ``x`` (for the soft penalty).
    :param lr: Adam learning rate.
    :param steps: Number of optimisation steps.
    :returns: ``(trained_model, loss_history)``.
    """
    graphdef, params, nontrain = nnx.split(model, nnx.Param, ...)  # type: ignore[misc]

    xc = jnp.asarray(data.collocation[:, 0])
    tc = jnp.asarray(data.collocation[:, 1])
    supervised = {
        name: (jnp.asarray(c), jnp.asarray(v))
        for name, pair in (("ic", data.ic), ("bc", data.bc), ("obs", data.obs))
        if pair is not None
        for c, v in [pair]
    }
    term_weight = {"ic": weights.ic, "bc": weights.bc, "obs": weights.data}

    def apply(p: Any, x: jax.Array, t: jax.Array) -> jax.Array:
        out: jax.Array = nnx.merge(graphdef, p, nontrain)(x, t)
        return out

    def scalar_u(p: Any, x: jax.Array, t: jax.Array) -> jax.Array:
        return apply(p, x.reshape(1, 1), t.reshape(1, 1))[0, 0]

    du_dx = jax.grad(scalar_u, argnums=1)
    du_dt = jax.grad(scalar_u, argnums=2)

    def loss_fn(p: Any) -> jax.Array:
        u = jax.vmap(scalar_u, in_axes=(None, 0, 0))(p, xc, tc)
        u_x = jax.vmap(du_dx, in_axes=(None, 0, 0))(p, xc, tc)
        u_t = jax.vmap(du_dt, in_axes=(None, 0, 0))(p, xc, tc)
        residual = u_t + problem.flux_prime(u) * u_x  # type: ignore[attr-defined]
        loss = weights.residual * jnp.mean(residual**2)
        if weights.mono > 0.0:
            loss += weights.mono * jnp.mean(jnp.maximum(-sign_x * u_x, 0.0) ** 2)
        for name, (coords, values) in supervised.items():
            pred = apply(p, coords[:, 0:1], coords[:, 1:2]).ravel()
            loss += term_weight[name] * jnp.mean((pred - values) ** 2)
        return loss

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(p: Any, o: Any) -> tuple[Any, Any, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(p)
        updates, o = optimizer.update(grads, o, p)
        return optax.apply_updates(p, updates), o, loss

    history: list[float] = []
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state)
        history.append(float(loss))

    return nnx.merge(graphdef, params, nontrain), history
