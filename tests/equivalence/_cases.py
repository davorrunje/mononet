"""Schema and loader for committed cross-backend equivalence vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_CASES_DIR = Path(__file__).parent / "cases"


@dataclass(frozen=True)
class EquivalenceCase:
    """One committed equivalence vector."""

    name: str
    kind: str
    inputs: dict[str, Any]
    params: dict[str, Any]
    expected_output: Any
    expected_grads: dict[str, Any]
    atol: float
    rtol: float

    def array(self, key: str, dtype: str = "float64") -> npt.NDArray[np.floating]:
        """Return an input array by key, as the case's dtype.

        :param key: Key into `self.inputs`.
        :param dtype: Target numpy dtype string.
        :returns: Numpy array with the requested dtype.
        """
        return np.asarray(self.inputs[key], dtype=dtype)

    @property
    def dtype(self) -> str:
        """Numpy dtype string for this case."""
        return str(self.params.get("dtype", "float64"))


def load_cases(kind: str) -> list[EquivalenceCase]:
    """Load all committed cases for a kind, sorted by filename.

    :param kind: `mono_linear`, `mono_residual`, or `mono_input`.
    :returns: List of `EquivalenceCase`.
    """
    out: list[EquivalenceCase] = []
    for path in sorted((_CASES_DIR / kind).glob("*.json")):
        data = json.loads(path.read_text())
        out.append(EquivalenceCase(kind=kind, **data))
    return out
