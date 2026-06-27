"""PyTorch idiomatic layer wrappers around mononet kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import nn

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask
from mononet.torch import _kernels

if TYPE_CHECKING:
    from collections.abc import Callable

_InitScheme = Literal["he_normal", "glorot_uniform", "lecun_normal"]

_INIT_FNS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "he_normal": lambda t: nn.init.kaiming_normal_(t, nonlinearity="relu"),
    "glorot_uniform": nn.init.xavier_uniform_,
    "lecun_normal": lambda t: nn.init.kaiming_normal_(t, nonlinearity="linear"),
}


def _act_name(activation: ActivationSpec | str) -> str:
    return activation if isinstance(activation, str) else activation.name


def _init_weight(weight: torch.Tensor, init: InitSpec | str | None) -> None:
    """Apply the requested initializer in-place.

    :param weight: Tensor to initialise.
    :param init: Initializer name, `InitSpec`, or `None` (defaults to
        `he_normal`).
    """
    if init is None:
        spec = InitSpec()
    elif isinstance(init, str):
        spec = InitSpec(scheme=init)  # type: ignore[arg-type]
    else:
        spec = init
    if spec.seed is not None:
        torch.manual_seed(spec.seed)
    _INIT_FNS[spec.scheme](weight)


class MonoLinear(nn.Module):
    """Monotonic analogue of `torch.nn.Linear` (non-decreasing in all inputs).

    :param in_features: Number of input features.
    :param units: Number of output features.
    :param mode: `"switch"` or `"absolute"`.
    :param activation: Base activation name or `ActivationSpec`.
    :param convex_fraction: Convex-neuron fraction (absolute mode).
    :param init: Weight initializer name/`InitSpec`/`None` (default `he_normal`).
    :param bias: Whether to include a bias term.
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        *,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        convex_fraction: float = 0.5,
        init: InitSpec | str | None = None,
        bias: bool = True,
    ) -> None:
        """Initialise MonoLinear."""
        super().__init__()
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.convex_fraction = convex_fraction
        self.weight = nn.Parameter(torch.empty(in_features, units))
        _init_weight(self.weight, init)
        self.bias = nn.Parameter(torch.zeros(units)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the monotonic dense transformation."""
        bias: torch.Tensor = (
            self.bias
            if self.bias is not None
            else torch.zeros(self.weight.shape[1], dtype=x.dtype, device=x.device)
        )
        return _kernels.monotonic_dense(
            x, self.weight, bias, self.mode, self.activation_name, self.convex_fraction
        )


class MonoResidual(nn.Module):
    """Dual-gated monotone residual block.

    :param in_features: Input feature count.
    :param units: Output feature count.
    :param F: Monotone sub-module, a `units -> Module` factory, or `None`
        (default: a single `MonoLinear`). A custom `F` carries the caller's
        responsibility for monotonicity.
    :param mode: Mode for the default `F`.
    :param activation: Activation for the default `F`.
    :param alpha_gate: Skip-gate token.
    :param beta_gate: Residual-gate token.
    :param init: Initializer for the default `F` and the projection.
    """

    def __init__(
        self,
        in_features: int,
        units: int,
        *,
        F: nn.Module | Callable[[int], nn.Module] | None = None,  # noqa: N803
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "scaled_elu",
        init: InitSpec | str | None = None,
    ) -> None:
        """Initialise MonoResidual."""
        super().__init__()
        if F is None:
            self.F: nn.Module = MonoLinear(
                in_features, units, mode=mode, activation=activation, init=init
            )
        elif callable(F) and not isinstance(F, nn.Module):
            self.F = F(units)
        else:
            self.F = F
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))
        if in_features == units:
            self.skip_weight: nn.Parameter | None = None
        else:
            sw = torch.empty(in_features, units)
            _init_weight(sw, init)
            self.skip_weight = nn.Parameter(sw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `g_alpha * skip(x) + g_beta * F(x)`."""
        skip: torch.Tensor = (
            x if self.skip_weight is None else x @ torch.exp(self.skip_weight)
        )
        fx: torch.Tensor = self.F(x)
        return (
            _kernels.gate(self.alpha_gate, self.alpha) * skip
            + _kernels.gate(self.beta_gate, self.beta) * fx
        )


class MonoInput(nn.Module):
    """Sign-flip layer mapping prescribed directions onto non-decreasing layers.

    :param directions: `+1`, `-1`, or a `MonotonicityMask` of per-feature
        `{-1,+1}` values.
    """

    def __init__(self, directions: int | MonotonicityMask) -> None:
        """Initialise MonoInput."""
        super().__init__()
        if isinstance(directions, MonotonicityMask):
            d = directions.values.astype(np.float32)
        else:
            d = np.array(float(directions), dtype=np.float32)
        self.register_buffer("directions", torch.tensor(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Negate `-1` columns; pass `+1` columns through."""
        dirs: torch.Tensor = self.directions  # type: ignore[assignment]
        return x * dirs.to(x.dtype)
