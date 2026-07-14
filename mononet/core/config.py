# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic configuration objects.

Plain dataclasses with `__post_init__` validation. Round-trip to JSON for
benchmark reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from mononet.core.types import ActivationSpec, InitSpec

Mode = Literal["mixed", "alternate", "split"]

_RENAMED_MODES = {"absolute": "mixed", "switch": "split"}


@dataclass(frozen=True, slots=True)
class MonoConfig:
    """Hyperparameters for a single monotonic dense layer.

    :param units: Number of output units; must be positive.
    :param mode: Construction mode — `"mixed"` (the paper's `|W|`
        construction, default), `"alternate"` (composition-aware alternating
        convex/concave layers), or `"split"` (the activation-switch variant).
    :param activation: Base activation applied by the layer (default
        `identity`).
    :param convex_fraction: Fraction of output units with a convex activation
        (mixed mode); must be in `[0, 1]`.
    :param init: Weight-initialization spec.
    :param bias: Whether the layer includes a bias term.
    :raises ValueError: If `units` is not positive, `mode` is unknown, or
        `convex_fraction` is outside `[0, 1]`.
    """

    units: int
    mode: Mode = "mixed"
    activation: ActivationSpec = field(
        default_factory=lambda: ActivationSpec("identity")
    )
    convex_fraction: float = 0.5
    init: InitSpec = field(default_factory=InitSpec)
    bias: bool = True

    def __post_init__(self) -> None:
        """Validate units, mode, and convex_fraction."""
        if self.units <= 0:
            raise ValueError(f"units must be positive; got {self.units}")
        if self.mode in _RENAMED_MODES:
            raise ValueError(
                f"mode {self.mode!r} was renamed; use "
                f"{_RENAMED_MODES[self.mode]!r} instead"
            )
        if self.mode not in ("mixed", "alternate", "split"):
            raise ValueError(
                f"mode must be 'mixed', 'alternate', or 'split'; got {self.mode!r}"
            )
        if not 0.0 <= self.convex_fraction <= 1.0:
            raise ValueError(
                f"convex_fraction must be in [0, 1]; got {self.convex_fraction}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain-Python dict suitable for JSON encoding."""
        return {
            "units": self.units,
            "mode": self.mode,
            "activation": {"name": self.activation.name},
            "convex_fraction": self.convex_fraction,
            "init": {"scheme": self.init.scheme, "seed": self.init.seed},
            "bias": self.bias,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonoConfig:
        """Deserialize from a plain-Python dict."""
        return cls(
            units=int(data["units"]),
            mode=data["mode"],
            activation=ActivationSpec(name=data["activation"]["name"]),
            convex_fraction=float(data["convex_fraction"]),
            init=InitSpec(scheme=data["init"]["scheme"], seed=data["init"]["seed"]),
            bias=bool(data["bias"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> MonoConfig:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True, slots=True)
class MonoResidualConfig:
    """Hyperparameters for a dual-gated monotonic residual block.

    Gate fields are string tokens only; a custom callable gate or `F`
    module is not serialized.

    :param units: Number of output units; must be positive.
    :param mode: Construction mode for the default `F` — `"mixed"`
        (default), `"alternate"`, or `"split"`.
    :param activation: Base activation for the default `F`. Required
        (keyword-only, no default) since a custom `F` is not representable
        here.
    :param alpha_gate: Gate token for the skip path.
    :param beta_gate: Gate token for the residual (transform) path.
    :param init: Weight-initialization spec.
    :param near_zero_scale: Scale applied to the default `F`'s last-layer
        weight (bias zeroed) so the block starts near-identity. `0.0`
        reproduces exact-zero (not recommended — see `MonoResidual`).
    :raises ValueError: If `units` is not positive or `mode` is unknown.
    """

    units: int
    mode: Mode = "mixed"
    activation: ActivationSpec = field(kw_only=True)
    alpha_gate: str = "shifted_elu"
    beta_gate: str = "softplus"
    init: InitSpec = field(default_factory=InitSpec)
    near_zero_scale: float = 1e-3

    def __post_init__(self) -> None:
        """Validate units and mode."""
        if self.units <= 0:
            raise ValueError(f"units must be positive; got {self.units}")
        if self.mode in _RENAMED_MODES:
            raise ValueError(
                f"mode {self.mode!r} was renamed; use "
                f"{_RENAMED_MODES[self.mode]!r} instead"
            )
        if self.mode not in ("mixed", "alternate", "split"):
            raise ValueError(
                f"mode must be 'mixed', 'alternate', or 'split'; got {self.mode!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain-Python dict suitable for JSON encoding."""
        return {
            "units": self.units,
            "mode": self.mode,
            "activation": {"name": self.activation.name},
            "alpha_gate": self.alpha_gate,
            "beta_gate": self.beta_gate,
            "init": {"scheme": self.init.scheme, "seed": self.init.seed},
            "near_zero_scale": self.near_zero_scale,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonoResidualConfig:
        """Deserialize from a plain-Python dict."""
        return cls(
            units=int(data["units"]),
            mode=data["mode"],
            activation=ActivationSpec(name=data["activation"]["name"]),
            alpha_gate=data["alpha_gate"],
            beta_gate=data["beta_gate"],
            init=InitSpec(scheme=data["init"]["scheme"], seed=data["init"]["seed"]),
            near_zero_scale=data["near_zero_scale"],
        )

    @classmethod
    def from_json(cls, payload: str) -> MonoResidualConfig:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(payload))
