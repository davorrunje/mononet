"""Backend-agnostic configuration objects.

Plain dataclasses with `__post_init__` validation. Round-trip to JSON for
benchmark reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask


@dataclass(frozen=True, slots=True)
class MonoLinearConfig:
    """Hyperparameters for a single monotonic linear layer."""

    in_features: int
    out_features: int
    monotonicity: MonotonicityMask
    activation: ActivationSpec
    init: InitSpec

    def __post_init__(self) -> None:
        """Validate layer dimensions and mask length."""
        if self.in_features <= 0:
            raise ValueError(f"in_features must be positive; got {self.in_features}")
        if self.out_features <= 0:
            raise ValueError(f"out_features must be positive; got {self.out_features}")
        if len(self.monotonicity) != self.in_features:
            raise ValueError(
                f"mask length ({len(self.monotonicity)}) "
                f"must equal in_features ({self.in_features})"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain-Python dict suitable for JSON encoding."""
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "monotonicity": self.monotonicity.values.tolist(),
            "activation": {"name": self.activation.name},
            "init": {"scheme": self.init.scheme, "seed": self.init.seed},
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonoLinearConfig:
        """Deserialize from a plain-Python dict."""
        return cls(
            in_features=int(data["in_features"]),
            out_features=int(data["out_features"]),
            monotonicity=MonotonicityMask(
                np.asarray(data["monotonicity"], dtype=np.int8)
            ),
            activation=ActivationSpec(name=data["activation"]["name"]),
            init=InitSpec(scheme=data["init"]["scheme"], seed=data["init"]["seed"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> MonoLinearConfig:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(payload))
