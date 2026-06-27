"""Backend-agnostic configuration objects.

Plain dataclasses with `__post_init__` validation. Round-trip to JSON for
benchmark reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from mononet.core.types import ActivationSpec, InitSpec

Mode = Literal["switch", "absolute"]


@dataclass(frozen=True, slots=True)
class MonoConfig:
    """Hyperparameters for a single monotonic dense layer."""

    units: int
    mode: Mode = "switch"
    activation: ActivationSpec = field(default_factory=lambda: ActivationSpec("relu"))
    convex_fraction: float = 0.5
    init: InitSpec = field(default_factory=InitSpec)
    bias: bool = True

    def __post_init__(self) -> None:
        """Validate units, mode, and convex_fraction."""
        if self.units <= 0:
            raise ValueError(f"units must be positive; got {self.units}")
        if self.mode not in ("switch", "absolute"):
            raise ValueError(f"mode must be 'switch' or 'absolute'; got {self.mode!r}")
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
    """

    units: int
    mode: Mode = "switch"
    activation: ActivationSpec = field(default_factory=lambda: ActivationSpec("relu"))
    alpha_gate: str = "shifted_elu"
    beta_gate: str = "scaled_elu"
    init: InitSpec = field(default_factory=InitSpec)

    def __post_init__(self) -> None:
        """Validate units and mode."""
        if self.units <= 0:
            raise ValueError(f"units must be positive; got {self.units}")
        if self.mode not in ("switch", "absolute"):
            raise ValueError(f"mode must be 'switch' or 'absolute'; got {self.mode!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain-Python dict suitable for JSON encoding."""
        return {
            "units": self.units,
            "mode": self.mode,
            "activation": {"name": self.activation.name},
            "alpha_gate": self.alpha_gate,
            "beta_gate": self.beta_gate,
            "init": {"scheme": self.init.scheme, "seed": self.init.seed},
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
        )

    @classmethod
    def from_json(cls, payload: str) -> MonoResidualConfig:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(payload))
