# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic model configuration and the build dispatcher.

A PINN model is a callable ``u(x, t)`` mapping two column vectors to one. The
four *methods* differ only in architecture (the loss is identical across them,
except the soft baseline's penalty, which lives in the trainer):

- ``vanilla``       — unconstrained MLP.
- ``soft``          — same unconstrained MLP; monotonicity via a loss penalty.
- ``weight_clip``   — inexpressive hard-monotone net (non-negative weights).
- ``hard_monotone`` — expressive hard-monotone net built from ``mononet`` layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mononet.core.config import Mode

Method = Literal["vanilla", "soft", "weight_clip", "hard_monotone"]
Backend = Literal["torch", "jax"]


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture hyperparameters shared by all methods and backends.

    Kept deliberately shallow (a few ``MonoResidual`` blocks): the repo's
    depth-vs-scale finding is that width, not depth, is the capacity lever.

    :param width: Hidden width of the monotone / plain stack.
    :param n_blocks: Number of residual blocks (``~4`` layers total with the
        head).
    :param t_embed_dim: Dimension of the unconstrained ``t`` embedding that makes
        the field free (non-monotone) in ``t``.
    :param t_embed_width: Hidden width of the ``t`` embedding MLP.
    :param mono_activation: Smooth activation for the monotone stack.
    :param plain_activation: Activation for unconstrained sub-networks.
    :param mode: ``mononet`` construction mode (``"absolute"`` or ``"switch"``).
    :param seed: Seed for parameter initialisation.
    """

    width: int = 32
    n_blocks: int = 2
    t_embed_dim: int = 8
    t_embed_width: int = 32
    mono_activation: str = "softplus"
    plain_activation: str = "tanh"
    mode: Mode = "absolute"
    seed: int = 0


def build(
    problem: object, cfg: ModelConfig, method: Method, backend: Backend
) -> object:
    """Dispatch to the backend-specific model builder.

    :param problem: A registered `Problem` (supplies the ``x`` monotonicity sign).
    :param cfg: Architecture configuration.
    :param method: One of the four methods.
    :param backend: ``"torch"`` or ``"jax"``.
    :returns: A backend-native model callable ``u(x, t)``.
    :raises NotImplementedError: If the backend builder is not yet available.
    """
    if backend == "torch":
        from applications.pinn.models.torch import builders as torch_builders

        return torch_builders.build_torch(problem, cfg, method)
    if backend == "jax":
        from applications.pinn.models.jax import builders as jax_builders

        return jax_builders.build_jax(problem, cfg, method)
    raise NotImplementedError(f"backend {backend!r} builder not yet implemented")
