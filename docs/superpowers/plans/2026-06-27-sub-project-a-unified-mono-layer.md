# Sub-project A — Unified Monotonic Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the unified monotonic dense primitive (`MonoLinear`/`MonoDense`, modes `switch`/`absolute`), the dual-gated `MonoResidual` block, and the `MonoInput` sign-flip layer across NumPy (reference), PyTorch, JAX (Flax NNX), and Keras 3, validated by a committed cross-backend equivalence harness.

**Architecture:** A stateless NumPy reference in `mononet.core.reference` is the arithmetic ground truth. Each backend has a pure-functional `_kernels.py` mirroring it one-to-one and idiomatic `layers.py` wrappers. Committed JSON test vectors (generated once from the reference) drive `tests/equivalence/`, asserting every backend matches the reference in output and gradient. Hypothesis property tests assert the non-decreasing guarantee per backend.

**Tech Stack:** Python 3.11+, NumPy, PyTorch ≥2.4, JAX ≥0.4.30 + Flax NNX ≥0.10, Keras ≥3.5 (JAX backend in CI), pytest, Hypothesis, mypy (strict), ruff.

## Global Constraints

- Python 3.11+; line length 88 (ruff); `target-version = py311`.
- Strict mypy throughout; type hints on every function/method.
- **MyST field-list docstrings** on all public functions/classes (`:param x: …`, `:returns: …`, `:raises X: …`); types come from annotations, never `:type:`/`:rtype:`.
- Stdlib `dataclasses` only for value objects. **Do not reintroduce Pydantic.**
- **Lazy backend imports:** `import mononet` must not import torch/jax/keras. Do not add backend imports to the top-level `__init__.py`.
- All monotonic constructions are **non-decreasing in every input**; layers carry no monotonicity mask.
- `gelu` is pinned to the **tanh approximation** in every backend and the reference (so cross-backend equivalence holds): `0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))`.
- SELU constants: `alpha=1.6732632423543772`, `scale=1.0507009873554805`. ELU `alpha=1.0`. Gate `ε=1e-3`.
- Commits must be **signed** (SSH/Secretive) and made on a branch, never on `main`. Per-task commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- Run backend tests with `MONONET_TEST_BACKEND={torch|jax|keras} uv run pytest …`; uninstalled backends `importorskip`.
- Frozen public API after this lands: `mononet.{torch,jax}.{MonoLinear,MonoResidual,MonoInput}`, `mononet.keras.{MonoDense,MonoResidual,MonoInput}`.

---

## File Structure

**Created**
- `tests/equivalence/_cases.py` — `EquivalenceCase` schema + `load_cases()`.
- `tools/regenerate-cases.py` — regenerates committed JSON vectors from the reference.
- `tests/equivalence/cases/{mono_linear,mono_residual,mono_input}/*.json` — committed vectors.
- `tests/equivalence/test_mono_linear.py`, `test_mono_residual.py`, `test_mono_input.py`.
- `tests/{torch,jax,keras}/test_property_monotonic.py` — Hypothesis property tests.

**Modified (rewritten)**
- `mononet/core/types.py` — `MonotonicityMask` → `{-1,+1}`; `ActivationSpec` → `Ă` family; `InitSpec` default `he_normal`.
- `mononet/core/config.py` — `MonoLinearConfig` → `MonoConfig` + `MonoResidualConfig`.
- `mononet/core/reference.py` — activations/gates + `monotonic_dense` (both modes) + `monotonic_residual`; drop `monotonic_mlp`.
- `mononet/core/__init__.py` — export configs.
- `mononet/{torch,jax,keras}/_kernels.py` — real kernels.
- `mononet/{torch,jax,keras}/layers.py` — `MonoLinear`/`MonoDense`, `MonoResidual`, `MonoInput`.
- `mononet/{torch,jax,keras}/__init__.py` — new exports.
- `mononet/__init__.py` — (no new backend imports; core re-exports unchanged).
- Existing tests under `tests/core/`, `tests/{torch,jax,keras}/test_public_api.py`, `tests/equivalence/test_placeholder.py`.
- `CLAUDE.md`, `docs/superpowers/specs/2026-05-21-mononet-package-design.md` — naming updates (§11).

**Deleted**
- `mononet/{torch,jax,keras}/models.py` (no `MonoMLP`/`MonoFeatureBlock`).

---

## Phase 1 — Core types & config (no backend dependencies)

### Task 1: `MonotonicityMask`, `ActivationSpec`, `InitSpec`

**Files:**
- Modify: `mononet/core/types.py`
- Test: `tests/core/test_types.py`

**Interfaces:**
- Produces: `MonotonicityMask(values: np.ndarray)` accepting only `{-1,+1}`; `ActivationName = Literal["relu","elu","selu","gelu","softplus"]`; `ActivationSpec(name)`; `InitSpec(scheme="he_normal", seed=None)` with `scheme ∈ {"he_normal","glorot_uniform","lecun_normal"}`.

- [ ] **Step 1: Rewrite the tests** in `tests/core/test_types.py` to the new contract.

```python
"""Unit tests for mononet.core.types."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask


class TestMonotonicityMask:
    def test_accepts_plus_minus_one(self) -> None:
        mask = MonotonicityMask(np.array([1, -1, 1, -1], dtype=np.int8))
        assert mask.shape == (4,)
        assert len(mask) == 4

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="must be in"):
            MonotonicityMask(np.array([1, 0, -1], dtype=np.int8))

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            MonotonicityMask(np.zeros((2, 3), dtype=np.int8))

    def test_is_frozen(self) -> None:
        mask = MonotonicityMask(np.array([1, -1], dtype=np.int8))
        with pytest.raises(dataclasses.FrozenInstanceError):
            mask.values = np.array([1, 1], dtype=np.int8)  # type: ignore[misc]


class TestActivationSpec:
    @pytest.mark.parametrize("name", ["relu", "elu", "selu", "gelu", "softplus"])
    def test_accepts_a_breve_family(self, name: str) -> None:
        assert ActivationSpec(name=name).name == name  # type: ignore[arg-type]

    @pytest.mark.parametrize("name", ["tanh", "sigmoid", "frobnicate"])
    def test_rejects_bounded_or_unknown(self, name: str) -> None:
        with pytest.raises(ValueError, match="unknown activation"):
            ActivationSpec(name=name)  # type: ignore[arg-type]


class TestInitSpec:
    def test_default_is_he_normal(self) -> None:
        assert InitSpec().scheme == "he_normal"
```

- [ ] **Step 2: Run, expect failure**

Run: `uv run pytest tests/core/test_types.py -q`
Expected: FAIL (mask accepts 0; `selu`/`gelu`/`softplus` rejected; default is `glorot_uniform`).

- [ ] **Step 3: Edit `mononet/core/types.py`**

Change the activation set and mask validation and init default:

```python
_KNOWN_ACTIVATIONS: frozenset[str] = frozenset(
    {"relu", "elu", "selu", "gelu", "softplus"}
)

ActivationName = Literal["relu", "elu", "selu", "gelu", "softplus"]
```

In `MonotonicityMask.__post_init__`, replace the membership check:

```python
        if not np.isin(arr, (-1, 1)).all():
            raise ValueError(
                "MonotonicityMask values must be in {-1, +1}; "
                f"got unique values {np.unique(arr).tolist()}"
            )
```

Update the `MonotonicityMask` class docstring to describe `{-1,+1}` (drop the `0` bullet). In `InitSpec`, change the default:

```python
    scheme: Literal["he_normal", "glorot_uniform", "lecun_normal"] = "he_normal"
    seed: int | None = None
```

- [ ] **Step 4: Run, expect pass**

Run: `uv run pytest tests/core/test_types.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mononet/core/types.py tests/core/test_types.py
git commit -m "feat(core): MonotonicityMask {-1,+1}, Ă activation family, he_normal default

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 2: `MonoConfig` and `MonoResidualConfig`

**Files:**
- Modify: `mononet/core/config.py` (replace `MonoLinearConfig`)
- Modify: `mononet/core/__init__.py`
- Test: `tests/core/test_config.py` (rewrite)

**Interfaces:**
- Produces:
  - `MonoConfig(units:int, mode:Literal["switch","absolute"]="switch", activation:ActivationSpec=ActivationSpec("relu"), convex_fraction:float=0.5, init:InitSpec=InitSpec(), bias:bool=True)` with `to_dict`/`from_dict`/`to_json`/`from_json`.
  - `MonoResidualConfig(units:int, mode:…="switch", activation:ActivationSpec=…, alpha_gate:str="shifted_elu", beta_gate:str="scaled_elu", init:InitSpec=…)` with the same serialization methods.

- [ ] **Step 1: Rewrite `tests/core/test_config.py`**

```python
"""Round-trip tests for mononet.core.config."""

from __future__ import annotations

import pytest

from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.types import ActivationSpec, InitSpec


def test_mono_config_roundtrip() -> None:
    cfg = MonoConfig(
        units=8,
        mode="absolute",
        activation=ActivationSpec("elu"),
        convex_fraction=0.25,
        init=InitSpec(scheme="he_normal", seed=3),
        bias=False,
    )
    assert MonoConfig.from_json(cfg.to_json()) == cfg


def test_mono_config_defaults() -> None:
    cfg = MonoConfig(units=4)
    assert cfg.mode == "switch"
    assert cfg.activation.name == "relu"
    assert cfg.convex_fraction == 0.5
    assert cfg.bias is True


def test_mono_config_rejects_bad_units_and_fraction() -> None:
    with pytest.raises(ValueError, match="units must be positive"):
        MonoConfig(units=0)
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoConfig(units=4, convex_fraction=1.5)


def test_mono_residual_config_roundtrip() -> None:
    cfg = MonoResidualConfig(
        units=16, mode="switch", activation=ActivationSpec("relu")
    )
    assert MonoResidualConfig.from_json(cfg.to_json()) == cfg
    assert cfg.alpha_gate == "shifted_elu"
    assert cfg.beta_gate == "scaled_elu"
```

- [ ] **Step 2: Run, expect failure** — `uv run pytest tests/core/test_config.py -q` (import error: no `MonoConfig`).

- [ ] **Step 3: Replace `mononet/core/config.py`**

```python
"""Backend-agnostic configuration objects.

Plain dataclasses with `__post_init__` validation. Round-trip to JSON for
benchmark reproducibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from mononet.core.types import ActivationSpec, InitSpec

Mode = Literal["switch", "absolute"]


@dataclass(frozen=True, slots=True)
class MonoConfig:
    """Hyperparameters for a single monotonic dense layer."""

    units: int
    mode: Mode = "switch"
    activation: ActivationSpec = ActivationSpec("relu")
    convex_fraction: float = 0.5
    init: InitSpec = InitSpec()
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
    activation: ActivationSpec = ActivationSpec("relu")
    alpha_gate: str = "shifted_elu"
    beta_gate: str = "scaled_elu"
    init: InitSpec = InitSpec()

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
```

- [ ] **Step 4: Update `mononet/core/__init__.py`**

```python
"""Framework-agnostic primitives shared by all backends."""

from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask

__all__ = [
    "ActivationSpec",
    "InitSpec",
    "MonoConfig",
    "MonoResidualConfig",
    "MonotonicityMask",
]
```

- [ ] **Step 5: Run, expect pass** — `uv run pytest tests/core/test_config.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add mononet/core/config.py mononet/core/__init__.py tests/core/test_config.py
git commit -m "feat(core): MonoConfig and MonoResidualConfig replace MonoLinearConfig

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2 — NumPy reference oracle

### Task 3: Activations, concave reflection, and gates

**Files:**
- Modify: `mononet/core/reference.py`
- Test: `tests/core/test_reference_activations.py` (create)

**Interfaces:**
- Produces (module-level in `reference.py`):
  - `base_activation(name: ActivationName, x: np.ndarray) -> np.ndarray` — `ρ̆`.
  - `concave_reflection(name: ActivationName, x: np.ndarray) -> np.ndarray` — `-ρ̆(-x)`.
  - `apply_gate(token: str, raw: np.ndarray) -> np.ndarray` — `shifted_elu`/`scaled_elu`.

- [ ] **Step 1: Create `tests/core/test_reference_activations.py`**

```python
"""Tests for the reference activations, reflection, and gates."""

from __future__ import annotations

import numpy as np
import pytest

from mononet.core import reference as ref


def test_relu_and_softplus() -> None:
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(ref.base_activation("relu", x), [0.0, 0.0, 3.0])
    np.testing.assert_allclose(
        ref.base_activation("softplus", x), np.log1p(np.exp(x)), atol=1e-6
    )


def test_concave_reflection_is_minus_act_of_minus_x() -> None:
    x = np.linspace(-3, 3, 7)
    np.testing.assert_allclose(
        ref.concave_reflection("relu", x), -ref.base_activation("relu", -x)
    )


@pytest.mark.parametrize("name", ["relu", "elu", "selu", "gelu", "softplus"])
def test_activations_are_nondecreasing(name: str) -> None:
    x = np.linspace(-5, 5, 200)
    y = ref.base_activation(name, x)  # type: ignore[arg-type]
    assert np.all(np.diff(y) >= -1e-7)


def test_gate_values_and_derivatives_at_zero() -> None:
    zero = np.array(0.0)
    assert ref.apply_gate("shifted_elu", zero) == pytest.approx(1.0)
    assert ref.apply_gate("scaled_elu", zero) == pytest.approx(1e-3)
    # finite-difference derivative at 0 is ~1 for both
    h = 1e-6
    for token in ("shifted_elu", "scaled_elu"):
        d = (
            ref.apply_gate(token, np.array(h)) - ref.apply_gate(token, np.array(-h))
        ) / (2 * h)
        assert d == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("token", ["shifted_elu", "scaled_elu"])
def test_gates_are_strictly_positive(token: str) -> None:
    x = np.linspace(-10, 10, 100)
    assert np.all(ref.apply_gate(token, x) > 0.0)
```

- [ ] **Step 2: Run, expect failure** — functions don't exist.

- [ ] **Step 3: Rewrite the top of `mononet/core/reference.py`**

Replace the module so it imports numpy eagerly and adds the helpers (keep the module docstring, update the paper line to cite both papers):

```python
"""NumPy reference implementations of the monotonic primitives.

Arithmetic ground truth for the cross-backend equivalence harness. Papers:
https://arxiv.org/abs/2205.11775 (absolute mode) and
https://arxiv.org/abs/2505.02537 (switch mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import numpy.typing as npt

    from mononet.core.types import ActivationName, ActivationSpec

_SELU_ALPHA = 1.6732632423543772
_SELU_SCALE = 1.0507009873554805
_GELU_C = 0.7978845608028654  # sqrt(2/pi)
_GATE_EPS = 1e-3


def base_activation(name: ActivationName, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Base activation `ρ̆ ∈ Ă` applied element-wise.

    :param name: One of `relu`, `elu`, `selu`, `gelu`, `softplus`.
    :param x: Input array.
    :returns: `ρ̆(x)`.
    """
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "elu":
        return np.where(x > 0.0, x, np.expm1(x))
    if name == "selu":
        return _SELU_SCALE * np.where(x > 0.0, x, _SELU_ALPHA * np.expm1(x))
    if name == "gelu":  # tanh approximation (pinned across backends)
        return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x**3)))
    if name == "softplus":
        return np.logaddexp(0.0, x)
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: ActivationName, x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Concave point reflection `ρ̂(x) = -ρ̆(-x)`.

    :param name: Base activation name.
    :param x: Input array.
    :returns: `ρ̂(x)`.
    """
    return -base_activation(name, -x)


def apply_gate(token: str, raw: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Resolve a gate string token and apply it to a raw parameter.

    :param token: `shifted_elu` (value 1, derivative 1 at 0) or `scaled_elu`
        (value `ε`, derivative 1 at 0).
    :param raw: Raw learnable gate parameter.
    :returns: A strictly-positive gate value.
    :raises ValueError: If the token is unknown.
    """
    if token == "shifted_elu":
        return np.where(raw > 0.0, raw, np.expm1(raw)) + 1.0
    if token == "scaled_elu":
        return np.maximum(raw, 0.0) + _GATE_EPS * np.exp(np.minimum(raw, 0.0) / _GATE_EPS)
    raise ValueError(f"unknown gate token {token!r}")
```

(Leave the `monotonic_dense`/`monotonic_mlp` stubs below for now; Task 4 replaces them.)

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/core/test_reference_activations.py -q`.

- [ ] **Step 5: Commit**

```bash
git add mononet/core/reference.py tests/core/test_reference_activations.py
git commit -m "feat(core): reference activations (Ă family), concave reflection, gates

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 4: `monotonic_dense` reference (both modes)

**Files:**
- Modify: `mononet/core/reference.py`
- Test: `tests/core/test_reference_signatures.py` (rewrite), `tests/core/test_reference_dense.py` (create)

**Interfaces:**
- Produces: `monotonic_dense(x, weights, bias, mode, activation, convex_fraction=0.5) -> np.ndarray`
  - `x: (B, n)`, `weights: (n, m)`, `bias: (m,)`, `mode: Literal["switch","absolute"]`, `activation: ActivationSpec`, `convex_fraction: float`. Returns `(B, m)`.

- [ ] **Step 1: Rewrite `tests/core/test_reference_signatures.py`**

```python
"""Tests that pin the public signature of the NumPy reference."""

from __future__ import annotations

import inspect

from mononet.core import reference


def test_monotonic_dense_signature() -> None:
    sig = inspect.signature(reference.monotonic_dense)
    assert list(sig.parameters) == [
        "x",
        "weights",
        "bias",
        "mode",
        "activation",
        "convex_fraction",
    ]


def test_monotonic_residual_exists() -> None:
    assert callable(reference.monotonic_residual)


def test_monotonic_mlp_removed() -> None:
    assert not hasattr(reference, "monotonic_mlp")
```

- [ ] **Step 2: Create `tests/core/test_reference_dense.py`**

```python
"""Numeric + monotonicity tests for monotonic_dense."""

from __future__ import annotations

import numpy as np
import pytest

from mononet.core import reference as ref
from mononet.core.types import ActivationSpec


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_switch_reduces_to_abs_weights_in_linear_regime() -> None:
    # With a piecewise-linear-at-large-scale check we instead verify the
    # identity at the formula level: σ=relu, large positive pre-activations.
    rng = _rng(0)
    x = rng.normal(size=(5, 3)).astype(np.float64)
    w = rng.normal(size=(3, 4)).astype(np.float64)
    b = np.zeros(4)
    y = ref.monotonic_dense(x, w, b, "switch", ActivationSpec("relu"), 0.5)
    assert y.shape == (5, 4)


@pytest.mark.parametrize("mode", ["switch", "absolute"])
@pytest.mark.parametrize("name", ["relu", "elu", "selu", "gelu", "softplus"])
def test_nondecreasing_in_every_input(mode: str, name: str) -> None:
    rng = _rng(7)
    w = rng.normal(size=(3, 5)).astype(np.float64)
    b = rng.normal(size=5).astype(np.float64)
    spec = ActivationSpec(name)  # type: ignore[arg-type]
    x = rng.normal(size=(8, 3)).astype(np.float64)
    bump = np.zeros_like(x)
    bump[:, 1] = 0.5  # increase input feature 1
    y0 = ref.monotonic_dense(x, w, b, mode, spec, 0.5)  # type: ignore[arg-type]
    y1 = ref.monotonic_dense(x + bump, w, b, mode, spec, 0.5)  # type: ignore[arg-type]
    assert np.all(y1 - y0 >= -1e-9)


def test_absolute_convex_fraction_endpoints() -> None:
    rng = _rng(1)
    x = rng.normal(size=(4, 2)).astype(np.float64)
    w = rng.normal(size=(2, 6)).astype(np.float64)
    b = rng.normal(size=6).astype(np.float64)
    spec = ActivationSpec("relu")
    h = x @ np.abs(w) + b
    all_convex = ref.monotonic_dense(x, w, b, "absolute", spec, 1.0)
    all_concave = ref.monotonic_dense(x, w, b, "absolute", spec, 0.0)
    np.testing.assert_allclose(all_convex, ref.base_activation("relu", h))
    np.testing.assert_allclose(all_concave, ref.concave_reflection("relu", h))
```

- [ ] **Step 3: Run, expect failure** — old `monotonic_dense` raises `NotImplementedError`.

- [ ] **Step 4: Replace the `monotonic_dense` stub in `reference.py`** and delete `monotonic_mlp`:

```python
def monotonic_dense(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    mode: str,
    activation: ActivationSpec,
    convex_fraction: float = 0.5,
) -> npt.NDArray[np.floating]:
    """Single monotonic dense transformation (NumPy reference).

    Non-decreasing in every input. `switch` uses the post-activation switch
    `σ(W⁺·x+b) − σ(W⁻·x+b)`; `absolute` uses `|W|·x+b` with the first
    `ceil(convex_fraction·m)` neurons convex and the rest concave.

    :param x: Input array of shape `(batch, in_features)`.
    :param weights: Weights of shape `(in_features, out_features)`.
    :param bias: Bias of shape `(out_features,)`.
    :param mode: `"switch"` or `"absolute"`.
    :param activation: Base activation `ρ̆`.
    :param convex_fraction: Convex-neuron fraction (absolute mode only).
    :returns: Output array of shape `(batch, out_features)`.
    :raises ValueError: If `mode` is not recognised.
    """
    name = activation.name
    if mode == "switch":
        w_pos = np.maximum(weights, 0.0)
        w_neg = np.minimum(weights, 0.0)
        return base_activation(name, x @ w_pos + bias) - base_activation(
            name, x @ w_neg + bias
        )
    if mode == "absolute":
        h = x @ np.abs(weights) + bias
        m = weights.shape[1]
        c = int(np.ceil(convex_fraction * m))
        out = np.empty_like(h)
        out[:, :c] = base_activation(name, h[:, :c])
        out[:, c:] = concave_reflection(name, h[:, c:])
        return out
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")
```

- [ ] **Step 5: Run, expect pass** — `uv run pytest tests/core/test_reference_signatures.py tests/core/test_reference_dense.py -q`.

- [ ] **Step 6: Commit**

```bash
git add mononet/core/reference.py tests/core/test_reference_signatures.py tests/core/test_reference_dense.py
git commit -m "feat(core): monotonic_dense reference for switch and absolute modes

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 5: `monotonic_residual` reference

**Files:**
- Modify: `mononet/core/reference.py`
- Test: `tests/core/test_reference_residual.py` (create)

**Interfaces:**
- Produces: `monotonic_residual(x, weights, bias, alpha, beta, *, mode="switch", activation, convex_fraction=0.5, alpha_gate="shifted_elu", beta_gate="scaled_elu", skip_weight=None) -> np.ndarray`
  - `weights/bias` are the inner mono-dense `F` params; `alpha`/`beta` are 0-d arrays; `skip_weight` is `(n, m)` or `None` (identity, requires `n==m`).

- [ ] **Step 1: Create `tests/core/test_reference_residual.py`**

```python
"""Tests for monotonic_residual reference."""

from __future__ import annotations

import numpy as np

from mononet.core import reference as ref
from mononet.core.types import ActivationSpec


def test_warm_start_is_near_identity() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 3)).astype(np.float64)
    w = rng.normal(size=(3, 3)).astype(np.float64)
    b = np.zeros(3)
    y = ref.monotonic_residual(
        x, w, b, np.array(0.0), np.array(0.0),
        mode="switch", activation=ActivationSpec("relu"),
    )
    # alpha gate = 1, beta gate = eps ≈ 0 → y ≈ identity skip
    np.testing.assert_allclose(y, x, atol=5e-3)


def test_projection_when_dims_differ() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(4, 2)).astype(np.float64)
    w = rng.normal(size=(2, 5)).astype(np.float64)
    b = np.zeros(5)
    skip = rng.normal(size=(2, 5)).astype(np.float64)
    y = ref.monotonic_residual(
        x, w, b, np.array(0.0), np.array(0.0),
        mode="switch", activation=ActivationSpec("relu"), skip_weight=skip,
    )
    assert y.shape == (4, 5)


def test_residual_is_nondecreasing() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6, 3)).astype(np.float64)
    w = rng.normal(size=(3, 3)).astype(np.float64)
    b = rng.normal(size=3).astype(np.float64)
    bump = np.zeros_like(x)
    bump[:, 0] = 0.4
    kw = dict(mode="switch", activation=ActivationSpec("relu"))
    y0 = ref.monotonic_residual(x, w, b, np.array(0.3), np.array(0.5), **kw)
    y1 = ref.monotonic_residual(x + bump, w, b, np.array(0.3), np.array(0.5), **kw)
    assert np.all(y1 - y0 >= -1e-9)
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Add `monotonic_residual` to `reference.py`**

```python
def monotonic_residual(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    alpha: npt.NDArray[np.floating],
    beta: npt.NDArray[np.floating],
    *,
    mode: str = "switch",
    activation: ActivationSpec,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.floating]:
    """Dual-gated monotone residual block (NumPy reference).

    `y = g_α(alpha)·skip(x) + g_β(beta)·F(x)`, with `F` a `monotonic_dense`
    and `skip` the identity (or `exp(skip_weight)` projection).

    :param x: Input `(batch, in_features)`.
    :param weights: `F` weights `(in_features, units)`.
    :param bias: `F` bias `(units,)`.
    :param alpha: Scalar raw skip-gate parameter.
    :param beta: Scalar raw residual-gate parameter.
    :param mode: `F` mode.
    :param activation: `F` base activation.
    :param convex_fraction: `F` convex fraction (absolute mode).
    :param alpha_gate: Skip-gate token.
    :param beta_gate: Residual-gate token.
    :param skip_weight: Projection weights `(in_features, units)`, or `None`
        for an identity skip (requires `in_features == units`).
    :returns: Output `(batch, units)`.
    """
    f = monotonic_dense(x, weights, bias, mode, activation, convex_fraction)
    skip = x if skip_weight is None else x @ np.exp(skip_weight)
    return apply_gate(alpha_gate, alpha) * skip + apply_gate(beta_gate, beta) * f
```

- [ ] **Step 4: Run, expect pass** — `uv run pytest tests/core/test_reference_residual.py -q`.

- [ ] **Step 5: Commit**

```bash
git add mononet/core/reference.py tests/core/test_reference_residual.py
git commit -m "feat(core): monotonic_residual reference (dual-gated block)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 3 — Equivalence harness infrastructure

### Task 6: Case schema and loader

**Files:**
- Create: `tests/equivalence/_cases.py`
- Delete: `tests/equivalence/test_placeholder.py`
- Test: `tests/equivalence/test_loader.py` (create)

**Interfaces:**
- Produces:
  - `@dataclass EquivalenceCase` with fields `name:str`, `kind:str` (`"mono_linear"|"mono_residual"|"mono_input"`), `inputs:dict[str, list]`, `params:dict[str, Any]`, `expected_output:list`, `expected_grads:dict[str, list]`, `atol:float`, `rtol:float`.
  - `arr(field) -> np.ndarray` helpers and `load_cases(kind: str) -> list[EquivalenceCase]` reading `tests/equivalence/cases/<kind>/*.json`.

- [ ] **Step 1: Create `tests/equivalence/_cases.py`**

```python
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
        """Return an input array by key, as the case's dtype."""
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
```

- [ ] **Step 2: Create `tests/equivalence/test_loader.py`** and remove the placeholder:

```python
"""Smoke test: the equivalence loader finds committed cases."""

from __future__ import annotations

import pytest

from tests.equivalence._cases import load_cases


@pytest.mark.parametrize("kind", ["mono_linear", "mono_residual", "mono_input"])
def test_cases_present(kind: str) -> None:
    cases = load_cases(kind)
    assert cases, f"no committed cases for {kind}"
```

```bash
git rm tests/equivalence/test_placeholder.py
```

- [ ] **Step 3: Run, expect failure** — no cases committed yet (loader returns empty). This passes only after Task 7. Leave failing; Task 7 makes it green.

- [ ] **Step 4: Commit the schema** (the loader test will go green after Task 7):

```bash
git add tests/equivalence/_cases.py tests/equivalence/test_loader.py
git commit -m "feat(equiv): EquivalenceCase schema and loader

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 7: Case generator + committed vectors

**Files:**
- Create: `tools/regenerate-cases.py`
- Create: `tests/equivalence/cases/{mono_linear,mono_residual,mono_input}/*.json`

**Interfaces:**
- Consumes: `mononet.core.reference.{monotonic_dense,monotonic_residual}`.
- Produces: committed JSON cases matching `EquivalenceCase` fields. Gradients are central finite differences over `sum(output)` w.r.t. each float param array, computed in float64.

- [ ] **Step 1: Create `tools/regenerate-cases.py`**

```python
"""Regenerate committed equivalence vectors from the NumPy reference.

Run: `uv run python tools/regenerate-cases.py`. Vectors are the source of
truth for tests/equivalence; CI never regenerates them.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mononet.core import reference as ref
from mononet.core.types import ActivationSpec

CASES = Path(__file__).resolve().parent.parent / "tests" / "equivalence" / "cases"
_FD_H = 1e-6
GRAD_ATOL, GRAD_RTOL = 1e-4, 1e-3
OUT_ATOL, OUT_RTOL = 1e-6, 1e-6


def _seed(name: str) -> int:
    """Deterministic per-case seed (stable across processes, unlike hash())."""
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")


def _fd_grad(f: Callable[[np.ndarray], np.ndarray], p: np.ndarray) -> np.ndarray:
    """Central finite-difference gradient of `sum(f(p))` w.r.t. `p`."""
    g = np.zeros_like(p)
    flat = p.ravel()
    gflat = g.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + _FD_H
        plus = float(f(p).sum())
        flat[i] = orig - _FD_H
        minus = float(f(p).sum())
        flat[i] = orig
        gflat[i] = (plus - minus) / (2 * _FD_H)
    return g


def _write(kind: str, name: str, payload: dict[str, Any]) -> None:
    d = CASES / kind
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


def _dense_cases() -> None:
    grid = [
        ("1x1x1", 1, 1, 1, "switch", "relu", 0.5),
        ("4x2x3-switch-relu", 4, 2, 3, "switch", "relu", 0.5),
        ("8x7x12-switch-elu", 8, 7, 12, "switch", "elu", 0.5),
        ("4x2x3-abs-relu-c1", 4, 2, 3, "absolute", "relu", 1.0),
        ("4x2x3-abs-relu-c0", 4, 2, 3, "absolute", "relu", 0.0),
        ("3x5x11-abs-selu", 3, 5, 11, "absolute", "selu", 0.5),
        ("8x7x12-switch-gelu", 8, 7, 12, "switch", "gelu", 0.5),
        ("2x16x1-switch-softplus", 2, 16, 1, "switch", "softplus", 0.5),
    ]
    for name, b, n, m, mode, act, cf in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        w = rng.normal(size=(n, m))
        bias = rng.normal(size=m)
        spec = ActivationSpec(act)  # type: ignore[arg-type]
        out = ref.monotonic_dense(x, w, bias, mode, spec, cf)
        gw = _fd_grad(lambda ww: ref.monotonic_dense(x, ww, bias, mode, spec, cf), w)
        gb = _fd_grad(lambda bb: ref.monotonic_dense(x, w, bb, mode, spec, cf), bias)
        _write("mono_linear", name, {
            "name": name,
            "inputs": {"x": x.tolist(), "weights": w.tolist(), "bias": bias.tolist()},
            "params": {"mode": mode, "activation": act, "convex_fraction": cf,
                       "dtype": "float64"},
            "expected_output": out.tolist(),
            "expected_grads": {"weights": gw.tolist(), "bias": gb.tolist()},
            "atol": OUT_ATOL, "rtol": OUT_RTOL,
        })


def _residual_cases() -> None:
    grid = [
        ("4x3x3-identity-switch", 4, 3, 3, None, "switch", "relu"),
        ("4x2x5-proj-switch", 4, 2, 5, (2, 5), "switch", "relu"),
        ("6x4x4-identity-abs", 6, 4, 4, None, "absolute", "elu"),
    ]
    for name, b, n, m, proj, mode, act in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        w = rng.normal(size=(n, m))
        bias = rng.normal(size=m)
        alpha = np.array(0.3)
        beta = np.array(0.5)
        skip = rng.normal(size=(n, m)) if proj else None
        spec = ActivationSpec(act)  # type: ignore[arg-type]

        def fwd(*, w=w, bias=bias, alpha=alpha, beta=beta, skip=skip) -> np.ndarray:
            return ref.monotonic_residual(
                x, w, bias, alpha, beta, mode=mode, activation=spec,
                skip_weight=skip,
            )

        out = fwd()
        grads = {
            "weights": _fd_grad(lambda v: fwd(w=v), w).tolist(),
            "bias": _fd_grad(lambda v: fwd(bias=v), bias).tolist(),
            "alpha": _fd_grad(lambda v: fwd(alpha=v), alpha).tolist(),
            "beta": _fd_grad(lambda v: fwd(beta=v), beta).tolist(),
        }
        inputs = {"x": x.tolist(), "weights": w.tolist(), "bias": bias.tolist(),
                  "alpha": alpha.tolist(), "beta": beta.tolist()}
        if skip is not None:
            inputs["skip_weight"] = skip.tolist()
            grads["skip_weight"] = _fd_grad(lambda v: fwd(skip=v), skip).tolist()
        _write("mono_residual", name, {
            "name": name,
            "inputs": inputs,
            "params": {"mode": mode, "activation": act, "convex_fraction": 0.5,
                       "alpha_gate": "shifted_elu", "beta_gate": "scaled_elu",
                       "has_projection": skip is not None, "dtype": "float64"},
            "expected_output": out.tolist(),
            "expected_grads": grads,
            "atol": OUT_ATOL, "rtol": GRAD_RTOL,
        })


def _input_cases() -> None:
    grid = [
        ("scalar-plus", 4, 3, [1, 1, 1]),
        ("scalar-minus", 4, 3, [-1, -1, -1]),
        ("mixed", 5, 4, [1, -1, 1, -1]),
    ]
    for name, b, n, directions in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        d = np.asarray(directions, dtype=np.float64)
        out = x * d
        _write("mono_input", name, {
            "name": name,
            "inputs": {"x": x.tolist()},
            "params": {"directions": directions, "dtype": "float64"},
            "expected_output": out.tolist(),
            "expected_grads": {},
            "atol": OUT_ATOL, "rtol": OUT_RTOL,
        })


def main() -> None:
    """Regenerate all case files and write a reference git hash."""
    _dense_cases()
    _residual_cases()
    _input_cases()
    sha = subprocess.run(
        ["git", "hash-object", "mononet/core/reference.py"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    (CASES / "REFERENCE_HASH").write_text(sha + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the vectors**

Run: `uv run python tools/regenerate-cases.py`
Expected: JSON files appear under `tests/equivalence/cases/{mono_linear,mono_residual,mono_input}/` plus `cases/REFERENCE_HASH`.

- [ ] **Step 3: Run the loader test, expect pass**

Run: `uv run pytest tests/equivalence/test_loader.py -q` → PASS.

- [ ] **Step 4: Commit generator + vectors**

```bash
git add tools/regenerate-cases.py tests/equivalence/cases
git commit -m "feat(equiv): case generator and committed vectors

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4 — PyTorch backend

### Task 8: PyTorch kernels + shared equivalence test for `mono_linear`

**Files:**
- Modify: `mononet/torch/_kernels.py`
- Create: `tests/equivalence/test_mono_linear.py`

**Interfaces:**
- Consumes: `EquivalenceCase`, `load_cases`.
- Produces (in `mononet/torch/_kernels.py`):
  - `activation(name: str, h: Tensor) -> Tensor`, `concave_reflection(name, h)`, `gate(token, raw)`.
  - `monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction) -> Tensor`.
  - `monotonic_residual(x, weights, bias, alpha, beta, *, mode, activation_name, convex_fraction, alpha_gate, beta_gate, skip_weight=None) -> Tensor`.

- [ ] **Step 1: Write `tests/equivalence/test_mono_linear.py`** (backend-parametrized via env)

```python
"""Cross-backend equivalence for the mono_linear kernel."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_linear")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch = pytest.importorskip("torch")
    from mononet.torch import _kernels as k

    x = torch.tensor(case.array("x"), dtype=torch.float64)
    w = torch.tensor(case.array("weights"), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(case.array("bias"), dtype=torch.float64, requires_grad=True)
    p = case.params
    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    y.sum().backward()
    return y.detach().numpy(), {"weights": w.grad.numpy(), "bias": b.grad.numpy()}


_RUNNERS = {"torch": _run_torch}  # jax/keras runners added in their phases


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_linear_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_linear runner for backend {BACKEND}")
    got, grads = runner(case)
    np.testing.assert_allclose(
        got, np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
    for key, expected in case.expected_grads.items():
        np.testing.assert_allclose(
            grads[key], np.asarray(expected), atol=1e-4, rtol=1e-3
        )
```

- [ ] **Step 2: Run, expect failure** — kernel raises `NotImplementedError`.

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/equivalence/test_mono_linear.py -q`

- [ ] **Step 3: Replace `mononet/torch/_kernels.py`**

```python
"""Private PyTorch kernels for monotonic primitives (stateless)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

_SELU = torch.nn.SELU()


def activation(name: str, h: torch.Tensor) -> torch.Tensor:
    """Apply the base activation `ρ̆` by name."""
    if name == "relu":
        return torch.relu(h)
    if name == "elu":
        return F.elu(h)
    if name == "selu":
        return _SELU(h)
    if name == "gelu":
        return F.gelu(h, approximate="tanh")
    if name == "softplus":
        return F.softplus(h)
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: str, h: torch.Tensor) -> torch.Tensor:
    """Concave reflection `ρ̂(h) = -ρ̆(-h)`."""
    return -activation(name, -h)


def gate(token: str, raw: torch.Tensor) -> torch.Tensor:
    """Resolve and apply a gate token to a raw parameter."""
    if token == "shifted_elu":
        return F.elu(raw) + 1.0
    if token == "scaled_elu":
        eps = 1e-3
        return torch.clamp(raw, min=0.0) + eps * torch.exp(torch.clamp(raw, max=0.0) / eps)
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
) -> torch.Tensor:
    """PyTorch monotonic dense kernel (see core.reference.monotonic_dense)."""
    if mode == "switch":
        w_pos = torch.clamp(weights, min=0.0)
        w_neg = torch.clamp(weights, max=0.0)
        return activation(activation_name, x @ w_pos + bias) - activation(
            activation_name, x @ w_neg + bias
        )
    if mode == "absolute":
        h = x @ torch.abs(weights) + bias
        m = weights.shape[1]
        import math

        c = int(math.ceil(convex_fraction * m))
        left = activation(activation_name, h[:, :c])
        right = concave_reflection(activation_name, h[:, c:])
        return torch.cat([left, right], dim=1)
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """PyTorch dual-gated monotone residual kernel."""
    f = monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction)
    skip = x if skip_weight is None else x @ torch.exp(skip_weight)
    return gate(alpha_gate, alpha) * skip + gate(beta_gate, beta) * f
```

- [ ] **Step 4: Run, expect pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/equivalence/test_mono_linear.py -q` → PASS.

- [ ] **Step 5: Commit**

```bash
git add mononet/torch/_kernels.py tests/equivalence/test_mono_linear.py
git commit -m "feat(torch): monotonic_dense/residual kernels + mono_linear equivalence

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 9: PyTorch `MonoLinear`, `MonoResidual`, `MonoInput` layers

**Files:**
- Modify: `mononet/torch/layers.py`
- Delete: `mononet/torch/models.py`
- Modify: `mononet/torch/__init__.py`
- Test: `tests/torch/test_public_api.py` (rewrite), `tests/equivalence/test_mono_residual.py` + `test_mono_input.py` (create)

**Interfaces:**
- Produces:
  - `MonoLinear(in_features:int, units:int, *, mode="switch", activation="relu", convex_fraction=0.5, init=None, bias=True)` — `nn.Module`; `weight` param `(in_features, units)`; `forward(x)`.
  - `MonoResidual(in_features:int, units:int, *, F=None, mode="switch", activation="relu", alpha_gate="shifted_elu", beta_gate="scaled_elu", init=None)` — `nn.Module`; scalar `alpha`/`beta` params; optional `skip_weight` `(in_features, units)`.
  - `MonoInput(directions: int | MonotonicityMask)` — `nn.Module`; `directions` buffer.

- [ ] **Step 1: Rewrite `tests/torch/test_public_api.py`**

```python
"""Contract test for the mononet.torch public API surface."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_exports() -> None:
    import mononet.torch as t

    assert set(t.__all__) == {"MonoLinear", "MonoResidual", "MonoInput"}


def test_mono_linear_is_module_and_runs() -> None:
    from mononet.torch import MonoLinear

    layer = MonoLinear(3, 5, mode="switch")
    assert isinstance(layer, torch.nn.Module)
    y = layer(torch.zeros(2, 3))
    assert y.shape == (2, 5)


def test_mono_residual_warm_start_near_identity() -> None:
    from mononet.torch import MonoResidual

    block = MonoResidual(4, 4, mode="switch")
    x = torch.randn(3, 4)
    y = block(x)
    assert torch.allclose(y, x, atol=5e-3)


def test_mono_input_flips_signs() -> None:
    from mononet.core.types import MonotonicityMask
    from mononet.torch import MonoInput

    layer = MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(layer(x), torch.tensor([[1.0, -2.0, 3.0]]))
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Replace `mononet/torch/layers.py`**

```python
"""PyTorch idiomatic layer wrappers around mononet kernels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask
from mononet.torch import _kernels

if TYPE_CHECKING:
    from collections.abc import Callable

_INIT_FNS: dict[str, Callable[[torch.Tensor], None]] = {
    "he_normal": lambda t: nn.init.kaiming_normal_(t, nonlinearity="relu"),
    "glorot_uniform": nn.init.xavier_uniform_,
    "lecun_normal": lambda t: nn.init.kaiming_normal_(t, nonlinearity="linear"),
}


def _act_name(activation: ActivationSpec | str) -> str:
    return activation if isinstance(activation, str) else activation.name


def _init_weight(weight: torch.Tensor, init: InitSpec | str | None) -> None:
    spec = InitSpec() if init is None else (InitSpec(scheme=init) if isinstance(init, str) else init)
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
        super().__init__()
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.convex_fraction = convex_fraction
        self.weight = nn.Parameter(torch.empty(in_features, units))
        _init_weight(self.weight, init)
        self.bias = nn.Parameter(torch.zeros(units)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the monotonic dense transformation."""
        bias = self.bias if self.bias is not None else torch.zeros(
            self.weight.shape[1], dtype=x.dtype, device=x.device
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
        F: nn.Module | Callable[[int], nn.Module] | None = None,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "scaled_elu",
        init: InitSpec | str | None = None,
    ) -> None:
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
        """Apply `g_α·skip(x) + g_β·F(x)`."""
        skip = x if self.skip_weight is None else x @ torch.exp(self.skip_weight)
        return (
            _kernels.gate(self.alpha_gate, self.alpha) * skip
            + _kernels.gate(self.beta_gate, self.beta) * self.F(x)
        )


class MonoInput(nn.Module):
    """Sign-flip layer mapping prescribed directions onto non-decreasing layers.

    :param directions: `+1`, `-1`, or a `MonotonicityMask` of per-feature
        `{-1,+1}` values.
    """

    def __init__(self, directions: int | MonotonicityMask) -> None:
        super().__init__()
        if isinstance(directions, MonotonicityMask):
            d = directions.values.astype(np.float32)
        else:
            d = np.array(float(directions), dtype=np.float32)
        self.register_buffer("directions", torch.tensor(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Negate `-1` columns; pass `+1` columns through."""
        return x * self.directions.to(x.dtype)
```

- [ ] **Step 4: Delete `models.py` and update `mononet/torch/__init__.py`**

```bash
git rm mononet/torch/models.py
```

```python
"""PyTorch backend for mononet.

Imports `torch` eagerly — only loaded when the user explicitly
imports `mononet.torch`.
"""

from mononet.torch.layers import MonoInput, MonoLinear, MonoResidual

__all__ = ["MonoInput", "MonoLinear", "MonoResidual"]
```

- [ ] **Step 5: Create `tests/equivalence/test_mono_residual.py`**

```python
"""Cross-backend equivalence for the mono_residual kernel."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_residual")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    torch = pytest.importorskip("torch")
    from mononet.torch import _kernels as k

    p = case.params
    t = lambda key: torch.tensor(case.array(key), dtype=torch.float64, requires_grad=True)
    x = torch.tensor(case.array("x"), dtype=torch.float64)
    w, b = t("weights"), t("bias")
    alpha, beta = t("alpha"), t("beta")
    sw = t("skip_weight") if p["has_projection"] else None
    y = k.monotonic_residual(
        x, w, b, alpha, beta, mode=p["mode"], activation_name=p["activation"],
        convex_fraction=p["convex_fraction"], alpha_gate=p["alpha_gate"],
        beta_gate=p["beta_gate"], skip_weight=sw,
    )
    y.sum().backward()
    grads = {"weights": w.grad.numpy(), "bias": b.grad.numpy(),
             "alpha": alpha.grad.numpy(), "beta": beta.grad.numpy()}
    if sw is not None:
        grads["skip_weight"] = sw.grad.numpy()
    return y.detach().numpy(), grads


_RUNNERS = {"torch": _run_torch}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_residual_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_residual runner for backend {BACKEND}")
    got, grads = runner(case)
    np.testing.assert_allclose(
        got, np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
    for key, expected in case.expected_grads.items():
        np.testing.assert_allclose(grads[key], np.asarray(expected), atol=1e-4, rtol=1e-3)
```

- [ ] **Step 6: Create `tests/equivalence/test_mono_input.py`**

```python
"""Cross-backend equivalence for the MonoInput sign-flip."""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.equivalence._cases import EquivalenceCase, load_cases

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
CASES = load_cases("mono_input")
IDS = [c.name for c in CASES]


def _run_torch(case: EquivalenceCase) -> np.ndarray:
    torch = pytest.importorskip("torch")
    from mononet.core.types import MonotonicityMask
    from mononet.torch import MonoInput

    directions = case.params["directions"]
    layer = MonoInput(MonotonicityMask(np.asarray(directions, dtype=np.int8)))
    return layer(torch.tensor(case.array("x"), dtype=torch.float64)).detach().numpy()


_RUNNERS = {"torch": _run_torch}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_mono_input_matches_reference(case: EquivalenceCase) -> None:
    runner = _RUNNERS.get(BACKEND)
    if runner is None:
        pytest.skip(f"no mono_input runner for backend {BACKEND}")
    np.testing.assert_allclose(
        runner(case), np.asarray(case.expected_output), atol=case.atol, rtol=case.rtol
    )
```

- [ ] **Step 7: Run, expect pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch tests/equivalence -q` → PASS.

- [ ] **Step 8: Commit**

```bash
git add mononet/torch/layers.py mononet/torch/__init__.py tests/torch/test_public_api.py tests/equivalence/test_mono_residual.py tests/equivalence/test_mono_input.py
git rm mononet/torch/models.py
git commit -m "feat(torch): MonoLinear, MonoResidual, MonoInput layers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 10: PyTorch monotonicity property test

**Files:**
- Create: `tests/torch/test_property_monotonic.py`

**Interfaces:**
- Consumes: `mononet.torch.{MonoLinear,MonoResidual}`.

- [ ] **Step 1: Write the Hypothesis test**

```python
"""Property test: PyTorch layers are non-decreasing in every input."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from hypothesis import given, settings
from hypothesis import strategies as st

from mononet.torch import MonoLinear, MonoResidual


@settings(deadline=None, max_examples=50)
@given(
    mode=st.sampled_from(["switch", "absolute"]),
    seed=st.integers(0, 10_000),
)
def test_mono_linear_nondecreasing(mode: str, seed: int) -> None:
    torch.manual_seed(seed)
    layer = MonoLinear(3, 4, mode=mode)
    x = torch.randn(6, 3)
    bump = torch.zeros_like(x)
    bump[:, 1] = torch.rand(6).abs() + 0.1
    with torch.no_grad():
        delta = layer(x + bump) - layer(x)
    assert torch.all(delta >= -1e-5)


@settings(deadline=None, max_examples=30)
@given(seed=st.integers(0, 10_000))
def test_mono_residual_nondecreasing(seed: int) -> None:
    torch.manual_seed(seed)
    block = MonoResidual(3, 3, mode="switch")
    # perturb the gate raws away from the warm start
    with torch.no_grad():
        block.beta.add_(torch.rand(()) * 2)
        block.alpha.add_(torch.rand(()) * 2)
    x = torch.randn(6, 3)
    bump = torch.zeros_like(x)
    bump[:, 0] = torch.rand(6).abs() + 0.1
    with torch.no_grad():
        delta = block(x + bump) - block(x)
    assert torch.all(delta >= -1e-5)
```

- [ ] **Step 2: Run, expect pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_property_monotonic.py -q` → PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/torch/test_property_monotonic.py
git commit -m "test(torch): Hypothesis monotonicity property tests

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 5 — JAX (Flax NNX) backend

### Task 11: JAX kernels + add jax runner to equivalence tests

**Files:**
- Modify: `mononet/jax/_kernels.py`
- Modify: `tests/equivalence/test_mono_linear.py`, `test_mono_residual.py` (add jax runner)

**Interfaces:**
- Produces: same kernel functions as torch, on `jnp.ndarray`. `activation`, `concave_reflection`, `gate`, `monotonic_dense`, `monotonic_residual`.

- [ ] **Step 1: Add a jax runner to `tests/equivalence/test_mono_linear.py`**

Insert before `_RUNNERS`:

```python
def _run_jax(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from mononet.jax import _kernels as k

    p = case.params
    x = jnp.asarray(case.array("x"))
    w = jnp.asarray(case.array("weights"))
    b = jnp.asarray(case.array("bias"))

    def loss(w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return k.monotonic_dense(
            x, w, b, p["mode"], p["activation"], p["convex_fraction"]
        ).sum()

    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    gw, gb = jax.grad(loss, argnums=(0, 1))(w, b)
    return np.asarray(y), {"weights": np.asarray(gw), "bias": np.asarray(gb)}
```

and change `_RUNNERS = {"torch": _run_torch}` → `_RUNNERS = {"torch": _run_torch, "jax": _run_jax}`.

- [ ] **Step 2: Add the analogous `_run_jax` to `test_mono_residual.py`**

```python
def _run_jax(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from mononet.jax import _kernels as k

    p = case.params
    x = jnp.asarray(case.array("x"))
    args = {n: jnp.asarray(case.array(n)) for n in ("weights", "bias", "alpha", "beta")}
    if p["has_projection"]:
        args["skip_weight"] = jnp.asarray(case.array("skip_weight"))
    names = list(args)

    def fwd(*vals: jnp.ndarray) -> jnp.ndarray:
        kw = dict(zip(names, vals))
        sw = kw.pop("skip_weight", None)
        return k.monotonic_residual(
            x, kw["weights"], kw["bias"], kw["alpha"], kw["beta"],
            mode=p["mode"], activation_name=p["activation"],
            convex_fraction=p["convex_fraction"], alpha_gate=p["alpha_gate"],
            beta_gate=p["beta_gate"], skip_weight=sw,
        )

    y = fwd(*args.values())
    grads = jax.grad(lambda *v: fwd(*v).sum(), argnums=tuple(range(len(names))))(
        *args.values()
    )
    return np.asarray(y), {n: np.asarray(g) for n, g in zip(names, grads)}
```

and extend `_RUNNERS` with `"jax": _run_jax`.

- [ ] **Step 3: Run, expect failure** — `MONONET_TEST_BACKEND=jax uv run pytest tests/equivalence/test_mono_linear.py -q` (kernel raises `NotImplementedError`).

- [ ] **Step 4: Replace `mononet/jax/_kernels.py`**

```python
"""Private JAX kernels for monotonic primitives (pure-functional)."""

from __future__ import annotations

import math

import jax.nn as jnn
import jax.numpy as jnp


def activation(name: str, h: jnp.ndarray) -> jnp.ndarray:
    """Apply the base activation `ρ̆` by name."""
    if name == "relu":
        return jnn.relu(h)
    if name == "elu":
        return jnn.elu(h)
    if name == "selu":
        return jnn.selu(h)
    if name == "gelu":
        return jnn.gelu(h, approximate=True)
    if name == "softplus":
        return jnn.softplus(h)
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: str, h: jnp.ndarray) -> jnp.ndarray:
    """Concave reflection `ρ̂(h) = -ρ̆(-h)`."""
    return -activation(name, -h)


def gate(token: str, raw: jnp.ndarray) -> jnp.ndarray:
    """Resolve and apply a gate token to a raw parameter."""
    if token == "shifted_elu":
        return jnn.elu(raw) + 1.0
    if token == "scaled_elu":
        eps = 1e-3
        return jnp.maximum(raw, 0.0) + eps * jnp.exp(jnp.minimum(raw, 0.0) / eps)
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
) -> jnp.ndarray:
    """JAX monotonic dense kernel."""
    if mode == "switch":
        w_pos = jnp.maximum(weights, 0.0)
        w_neg = jnp.minimum(weights, 0.0)
        return activation(activation_name, x @ w_pos + bias) - activation(
            activation_name, x @ w_neg + bias
        )
    if mode == "absolute":
        h = x @ jnp.abs(weights) + bias
        c = int(math.ceil(convex_fraction * weights.shape[1]))
        left = activation(activation_name, h[:, :c])
        right = concave_reflection(activation_name, h[:, c:])
        return jnp.concatenate([left, right], axis=1)
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: jnp.ndarray,
    weights: jnp.ndarray,
    bias: jnp.ndarray,
    alpha: jnp.ndarray,
    beta: jnp.ndarray,
    *,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """JAX dual-gated monotone residual kernel."""
    f = monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction)
    skip = x if skip_weight is None else x @ jnp.exp(skip_weight)
    return gate(alpha_gate, alpha) * skip + gate(beta_gate, beta) * f
```

- [ ] **Step 5: Run, expect pass**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/equivalence/test_mono_linear.py tests/equivalence/test_mono_residual.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add mononet/jax/_kernels.py tests/equivalence/test_mono_linear.py tests/equivalence/test_mono_residual.py
git commit -m "feat(jax): monotonic_dense/residual kernels + equivalence runners

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 12: JAX `MonoLinear`, `MonoResidual`, `MonoInput` layers

**Files:**
- Modify: `mononet/jax/layers.py`
- Delete: `mononet/jax/models.py`
- Modify: `mononet/jax/__init__.py`
- Test: `tests/jax/test_public_api.py` (rewrite), `tests/jax/test_property_monotonic.py` (create), add `_run_jax` to `tests/equivalence/test_mono_input.py`

**Interfaces:**
- Produces (NNX modules; constructor takes `*, rngs: nnx.Rngs`):
  - `MonoLinear(in_features, units, *, mode="switch", activation="relu", convex_fraction=0.5, init=None, bias=True, rngs)`.
  - `MonoResidual(in_features, units, *, F=None, mode="switch", activation="relu", alpha_gate="shifted_elu", beta_gate="scaled_elu", init=None, rngs)`.
  - `MonoInput(directions)` — no rngs.

- [ ] **Step 1: Rewrite `tests/jax/test_public_api.py`**

```python
"""Contract test for the mononet.jax public API surface."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp


def test_exports() -> None:
    import mononet.jax as j

    assert set(j.__all__) == {"MonoLinear", "MonoResidual", "MonoInput"}


def test_mono_linear_runs() -> None:
    from mononet.jax import MonoLinear

    layer = MonoLinear(3, 5, mode="switch", rngs=nnx.Rngs(0))
    y = layer(jnp.zeros((2, 3)))
    assert y.shape == (2, 5)


def test_mono_residual_warm_start_near_identity() -> None:
    from mononet.jax import MonoResidual

    block = MonoResidual(4, 4, mode="switch", rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (3, 4))
    assert jnp.allclose(block(x), x, atol=5e-3)


def test_mono_input_flips_signs() -> None:
    from mononet.core.types import MonotonicityMask
    from mononet.jax import MonoInput

    layer = MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(layer(x), jnp.array([[1.0, -2.0, 3.0]]))
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Replace `mononet/jax/layers.py`**

```python
"""JAX (Flax NNX) idiomatic layer wrappers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.nn.initializers as jinit
import jax.numpy as jnp
import numpy as np
from flax import nnx

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask
from mononet.jax import _kernels

if TYPE_CHECKING:
    from collections.abc import Callable

_INIT_FNS = {
    "he_normal": jinit.he_normal(),
    "glorot_uniform": jinit.glorot_uniform(),
    "lecun_normal": jinit.lecun_normal(),
}


def _act_name(activation: ActivationSpec | str) -> str:
    return activation if isinstance(activation, str) else activation.name


def _init_array(shape: tuple[int, int], init: InitSpec | str | None, rngs: nnx.Rngs) -> jnp.ndarray:
    spec = InitSpec() if init is None else (InitSpec(scheme=init) if isinstance(init, str) else init)
    return _INIT_FNS[spec.scheme](rngs.params(), shape)


class MonoLinear(nnx.Module):
    """Monotonic analogue of `flax.nnx.Linear` (non-decreasing in all inputs)."""

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
        rngs: nnx.Rngs,
    ) -> None:
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.convex_fraction = convex_fraction
        self.weight = nnx.Param(_init_array((in_features, units), init, rngs))
        self.bias = nnx.Param(jnp.zeros((units,))) if bias else None

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the monotonic dense transformation."""
        bias = self.bias.value if self.bias is not None else jnp.zeros(
            (self.weight.value.shape[1],), dtype=x.dtype
        )
        return _kernels.monotonic_dense(
            x, self.weight.value, bias, self.mode, self.activation_name, self.convex_fraction
        )


class MonoResidual(nnx.Module):
    """Dual-gated monotone residual block (Flax NNX)."""

    def __init__(
        self,
        in_features: int,
        units: int,
        *,
        F: nnx.Module | Callable[[int], nnx.Module] | None = None,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "scaled_elu",
        init: InitSpec | str | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        if F is None:
            self.F: nnx.Module = MonoLinear(
                in_features, units, mode=mode, activation=activation, init=init, rngs=rngs
            )
        elif callable(F) and not isinstance(F, nnx.Module):
            self.F = F(units)
        else:
            self.F = F
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        self.alpha = nnx.Param(jnp.zeros(()))
        self.beta = nnx.Param(jnp.zeros(()))
        if in_features == units:
            self.skip_weight: nnx.Param | None = None
        else:
            self.skip_weight = nnx.Param(_init_array((in_features, units), init, rngs))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply `g_α·skip(x) + g_β·F(x)`."""
        skip = x if self.skip_weight is None else x @ jnp.exp(self.skip_weight.value)
        return (
            _kernels.gate(self.alpha_gate, self.alpha.value) * skip
            + _kernels.gate(self.beta_gate, self.beta.value) * self.F(x)
        )


class MonoInput(nnx.Module):
    """Sign-flip layer mapping prescribed directions onto non-decreasing layers."""

    def __init__(self, directions: int | MonotonicityMask) -> None:
        if isinstance(directions, MonotonicityMask):
            d = directions.values.astype(np.float32)
        else:
            d = np.array(float(directions), dtype=np.float32)
        self.directions = nnx.Variable(jnp.asarray(d))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Negate `-1` columns; pass `+1` columns through."""
        return x * self.directions.value.astype(x.dtype)
```

- [ ] **Step 4: Delete `models.py` and update `mononet/jax/__init__.py`**

```bash
git rm mononet/jax/models.py
```

```python
"""JAX backend (Flax NNX) for mononet."""

from mononet.jax.layers import MonoInput, MonoLinear, MonoResidual

__all__ = ["MonoInput", "MonoLinear", "MonoResidual"]
```

- [ ] **Step 5: Add `_run_jax` to `tests/equivalence/test_mono_input.py`**

```python
def _run_jax(case: EquivalenceCase) -> np.ndarray:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from mononet.core.types import MonotonicityMask
    from mononet.jax import MonoInput

    layer = MonoInput(MonotonicityMask(np.asarray(case.params["directions"], dtype=np.int8)))
    return np.asarray(layer(jnp.asarray(case.array("x"))))
```

and add `"jax": _run_jax` to `_RUNNERS`.

- [ ] **Step 6: Create `tests/jax/test_property_monotonic.py`**

```python
"""Property test: JAX layers are non-decreasing in every input."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp
from hypothesis import given, settings
from hypothesis import strategies as st

from mononet.jax import MonoLinear


@settings(deadline=None, max_examples=40)
@given(mode=st.sampled_from(["switch", "absolute"]), seed=st.integers(0, 10_000))
def test_mono_linear_nondecreasing(mode: str, seed: int) -> None:
    layer = MonoLinear(3, 4, mode=mode, rngs=nnx.Rngs(seed))
    key = jax.random.key(seed)
    x = jax.random.normal(key, (6, 3))
    bump = jnp.zeros_like(x).at[:, 1].set(0.5)
    delta = layer(x + bump) - layer(x)
    assert bool(jnp.all(delta >= -1e-5))
```

- [ ] **Step 7: Run, expect pass**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax tests/equivalence -q` → PASS.

- [ ] **Step 8: Commit**

```bash
git add mononet/jax/layers.py mononet/jax/__init__.py tests/jax tests/equivalence/test_mono_input.py
git rm mononet/jax/models.py
git commit -m "feat(jax): MonoLinear, MonoResidual, MonoInput layers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 6 — Keras 3 backend

### Task 13: Keras kernels + add keras runner to equivalence tests

**Files:**
- Modify: `mononet/keras/_kernels.py`
- Modify: `tests/equivalence/test_mono_linear.py`, `test_mono_residual.py` (add keras runner)

**Interfaces:**
- Produces: `activation`, `concave_reflection`, `gate`, `monotonic_dense`, `monotonic_residual` using `keras.ops`.

- [ ] **Step 1: Add `_run_keras` to `tests/equivalence/test_mono_linear.py`** (gradients via the active Keras backend — use `keras.ops` + backend autograd through a small helper; here use JAX-backed Keras grad through `keras.ops`):

```python
def _run_keras(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pytest.importorskip("keras")
    import jax  # CI default KERAS_BACKEND=jax
    import jax.numpy as jnp

    from mononet.keras import _kernels as k

    p = case.params
    x = jnp.asarray(case.array("x"))

    def loss(w: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
        ).sum()

    w = jnp.asarray(case.array("weights"))
    b = jnp.asarray(case.array("bias"))
    y = k.monotonic_dense(x, w, b, p["mode"], p["activation"], p["convex_fraction"])
    gw, gb = jax.grad(loss, argnums=(0, 1))(w, b)
    return np.asarray(y), {"weights": np.asarray(gw), "bias": np.asarray(gb)}
```

Add `"keras": _run_keras` to `_RUNNERS`. (Note: this assumes `KERAS_BACKEND=jax`, the CI default; a `keras`-backend grad test under torch is out of scope for equivalence and covered by the output-only public-API test.)

- [ ] **Step 2: Add the analogous `_run_keras` to `test_mono_residual.py`** mirroring `_run_jax` but importing `mononet.keras._kernels`. (Same structure; gradients via `jax.grad`.)

```python
def _run_keras(case: EquivalenceCase) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    pytest.importorskip("keras")
    import jax
    import jax.numpy as jnp

    from mononet.keras import _kernels as k

    p = case.params
    x = jnp.asarray(case.array("x"))
    args = {n: jnp.asarray(case.array(n)) for n in ("weights", "bias", "alpha", "beta")}
    if p["has_projection"]:
        args["skip_weight"] = jnp.asarray(case.array("skip_weight"))
    names = list(args)

    def fwd(*vals: jnp.ndarray) -> jnp.ndarray:
        kw = dict(zip(names, vals))
        sw = kw.pop("skip_weight", None)
        return jnp.asarray(k.monotonic_residual(
            x, kw["weights"], kw["bias"], kw["alpha"], kw["beta"],
            mode=p["mode"], activation_name=p["activation"],
            convex_fraction=p["convex_fraction"], alpha_gate=p["alpha_gate"],
            beta_gate=p["beta_gate"], skip_weight=sw,
        ))

    y = fwd(*args.values())
    grads = jax.grad(lambda *v: fwd(*v).sum(), argnums=tuple(range(len(names))))(*args.values())
    return np.asarray(y), {n: np.asarray(g) for n, g in zip(names, grads)}
```

Add `"keras": _run_keras` to `_RUNNERS`.

- [ ] **Step 3: Run, expect failure** — `MONONET_TEST_BACKEND=keras uv run pytest tests/equivalence/test_mono_linear.py -q`.

- [ ] **Step 4: Replace `mononet/keras/_kernels.py`**

```python
"""Private Keras 3 kernels for monotonic primitives (keras.ops only)."""

from __future__ import annotations

import math
from typing import Any

import keras
from keras import ops


def activation(name: str, h: Any) -> Any:
    """Apply the base activation `ρ̆` by name."""
    if name == "relu":
        return ops.relu(h)
    if name == "elu":
        return ops.elu(h)
    if name == "selu":
        return ops.selu(h)
    if name == "gelu":
        return ops.gelu(h, approximate=True)
    if name == "softplus":
        return ops.softplus(h)
    raise ValueError(f"unknown activation {name!r}")


def concave_reflection(name: str, h: Any) -> Any:
    """Concave reflection `ρ̂(h) = -ρ̆(-h)`."""
    return -activation(name, -h)


def gate(token: str, raw: Any) -> Any:
    """Resolve and apply a gate token to a raw parameter."""
    if token == "shifted_elu":
        return ops.elu(raw) + 1.0
    if token == "scaled_elu":
        eps = 1e-3
        return ops.maximum(raw, 0.0) + eps * ops.exp(ops.minimum(raw, 0.0) / eps)
    raise ValueError(f"unknown gate token {token!r}")


def monotonic_dense(
    x: Any,
    weights: Any,
    bias: Any,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
) -> Any:
    """Keras monotonic dense kernel."""
    if mode == "switch":
        w_pos = ops.maximum(weights, 0.0)
        w_neg = ops.minimum(weights, 0.0)
        return activation(activation_name, ops.matmul(x, w_pos) + bias) - activation(
            activation_name, ops.matmul(x, w_neg) + bias
        )
    if mode == "absolute":
        h = ops.matmul(x, ops.abs(weights)) + bias
        c = int(math.ceil(convex_fraction * int(weights.shape[1])))
        left = activation(activation_name, h[:, :c])
        right = concave_reflection(activation_name, h[:, c:])
        return ops.concatenate([left, right], axis=1)
    raise ValueError(f"mode must be 'switch' or 'absolute'; got {mode!r}")


def monotonic_residual(
    x: Any,
    weights: Any,
    bias: Any,
    alpha: Any,
    beta: Any,
    *,
    mode: str,
    activation_name: str,
    convex_fraction: float = 0.5,
    alpha_gate: str = "shifted_elu",
    beta_gate: str = "scaled_elu",
    skip_weight: Any | None = None,
) -> Any:
    """Keras dual-gated monotone residual kernel."""
    f = monotonic_dense(x, weights, bias, mode, activation_name, convex_fraction)
    skip = x if skip_weight is None else ops.matmul(x, ops.exp(skip_weight))
    return gate(alpha_gate, alpha) * skip + gate(beta_gate, beta) * f
```

- [ ] **Step 5: Run, expect pass**

Run: `MONONET_TEST_BACKEND=keras KERAS_BACKEND=jax uv run pytest tests/equivalence/test_mono_linear.py tests/equivalence/test_mono_residual.py -q` → PASS.

- [ ] **Step 6: Commit**

```bash
git add mononet/keras/_kernels.py tests/equivalence/test_mono_linear.py tests/equivalence/test_mono_residual.py
git commit -m "feat(keras): monotonic_dense/residual kernels + equivalence runners

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

### Task 14: Keras `MonoDense`, `MonoResidual`, `MonoInput` layers

**Files:**
- Modify: `mononet/keras/layers.py`
- Delete: `mononet/keras/models.py`
- Modify: `mononet/keras/__init__.py`
- Test: `tests/keras/test_public_api.py` (rewrite), `tests/keras/test_property_monotonic.py` (create), add `_run_keras` to `tests/equivalence/test_mono_input.py`

**Interfaces:**
- Produces:
  - `MonoDense(units, *, mode="switch", activation="relu", convex_fraction=0.5, init=None, bias=True, **kwargs)` — `keras.layers.Layer`; infers `in_features` in `build`; `get_config` round-trips token/scalar fields.
  - `MonoResidual(units, *, F=None, mode="switch", activation="relu", alpha_gate="shifted_elu", beta_gate="scaled_elu", init=None, **kwargs)`.
  - `MonoInput(directions, **kwargs)`.

- [ ] **Step 1: Rewrite `tests/keras/test_public_api.py`**

```python
"""Contract test for the mononet.keras public API surface."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops


def test_exports() -> None:
    import mononet.keras as kmod

    assert set(kmod.__all__) == {"MonoDense", "MonoResidual", "MonoInput"}


def test_mono_dense_runs_and_serializes() -> None:
    from mononet.keras import MonoDense

    layer = MonoDense(5, mode="absolute", convex_fraction=0.25)
    y = layer(ops.zeros((2, 3)))
    assert tuple(y.shape) == (2, 5)
    cfg = layer.get_config()
    clone = MonoDense.from_config(cfg)
    assert clone.mode == "absolute"
    assert clone.convex_fraction == 0.25


def test_mono_residual_warm_start_near_identity() -> None:
    from mononet.keras import MonoResidual

    block = MonoResidual(4, mode="switch")
    x = ops.convert_to_tensor(np.random.default_rng(0).normal(size=(3, 4)))
    assert bool(ops.all(ops.abs(block(x) - x) < 5e-3))


def test_mono_input_flips_signs() -> None:
    from mononet.core.types import MonotonicityMask
    from mononet.keras import MonoInput

    layer = MonoInput(MonotonicityMask(np.array([1, -1, 1], dtype=np.int8)))
    x = ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
    assert bool(ops.all(layer(x) == ops.convert_to_tensor(np.array([[1.0, -2.0, 3.0]]))))
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Replace `mononet/keras/layers.py`**

```python
"""Keras 3 idiomatic layer wrappers."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np
from keras import ops

from mononet.core.types import ActivationSpec, InitSpec, MonotonicityMask
from mononet.keras import _kernels

_INIT_NAMES = {
    "he_normal": "he_normal",
    "glorot_uniform": "glorot_uniform",
    "lecun_normal": "lecun_normal",
}


def _act_name(activation: ActivationSpec | str) -> str:
    return activation if isinstance(activation, str) else activation.name


def _init_name(init: InitSpec | str | None) -> str:
    if init is None:
        return "he_normal"
    return init if isinstance(init, str) else init.scheme


class MonoDense(keras.layers.Layer):  # type: ignore[misc]
    """Monotonic analogue of `keras.layers.Dense` (non-decreasing in all inputs)."""

    def __init__(
        self,
        units: int,
        *,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        convex_fraction: float = 0.5,
        init: InitSpec | str | None = None,
        bias: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.convex_fraction = convex_fraction
        self.init_name = _init_name(init)
        self.use_bias = bias

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Create weights once the input width is known."""
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer=self.init_name,
            trainable=True,
            name="weight",
        )
        self.b = (
            self.add_weight(shape=(self.units,), initializer="zeros", trainable=True, name="bias")
            if self.use_bias
            else None
        )

    def call(self, inputs: Any) -> Any:
        """Apply the monotonic dense transformation."""
        bias = self.b if self.b is not None else ops.zeros((self.units,))
        return _kernels.monotonic_dense(
            inputs, self.w, bias, self.mode, self.activation_name, self.convex_fraction
        )

    def get_config(self) -> dict[str, Any]:
        """Serialize token/scalar fields (callables are not serializable)."""
        cfg = super().get_config()
        cfg.update({
            "units": self.units,
            "mode": self.mode,
            "activation": self.activation_name,
            "convex_fraction": self.convex_fraction,
            "init": self.init_name,
            "bias": self.use_bias,
        })
        return cfg


class MonoResidual(keras.layers.Layer):  # type: ignore[misc]
    """Dual-gated monotone residual block (Keras 3)."""

    def __init__(
        self,
        units: int,
        *,
        F: keras.layers.Layer | None = None,
        mode: str = "switch",
        activation: ActivationSpec | str = "relu",
        alpha_gate: str = "shifted_elu",
        beta_gate: str = "scaled_elu",
        init: InitSpec | str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.mode = mode
        self.activation_name = _act_name(activation)
        self.init_name = _init_name(init)
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        self.F = F if F is not None else MonoDense(
            units, mode=mode, activation=activation, init=init
        )

    def build(self, input_shape: tuple[int, ...]) -> None:
        """Create gate scalars and the projection shortcut if needed."""
        in_features = int(input_shape[-1])
        self.alpha = self.add_weight(shape=(), initializer="zeros", trainable=True, name="alpha")
        self.beta = self.add_weight(shape=(), initializer="zeros", trainable=True, name="beta")
        self.skip_w = (
            None
            if in_features == self.units
            else self.add_weight(
                shape=(in_features, self.units), initializer=self.init_name,
                trainable=True, name="skip_weight",
            )
        )

    def call(self, inputs: Any) -> Any:
        """Apply `g_α·skip(x) + g_β·F(x)`."""
        skip = inputs if self.skip_w is None else ops.matmul(inputs, ops.exp(self.skip_w))
        return (
            _kernels.gate(self.alpha_gate, self.alpha) * skip
            + _kernels.gate(self.beta_gate, self.beta) * self.F(inputs)
        )

    def get_config(self) -> dict[str, Any]:
        """Serialize token/scalar fields."""
        cfg = super().get_config()
        cfg.update({
            "units": self.units, "mode": self.mode, "activation": self.activation_name,
            "alpha_gate": self.alpha_gate, "beta_gate": self.beta_gate, "init": self.init_name,
        })
        return cfg


class MonoInput(keras.layers.Layer):  # type: ignore[misc]
    """Sign-flip layer mapping prescribed directions onto non-decreasing layers."""

    def __init__(self, directions: int | MonotonicityMask, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if isinstance(directions, MonotonicityMask):
            self._directions = directions.values.astype(np.float32).tolist()
        else:
            self._directions = float(directions)

    def call(self, inputs: Any) -> Any:
        """Negate `-1` columns; pass `+1` columns through."""
        return inputs * ops.convert_to_tensor(self._directions, dtype=inputs.dtype)

    def get_config(self) -> dict[str, Any]:
        """Serialize the direction spec."""
        cfg = super().get_config()
        cfg.update({"directions": self._directions})
        return cfg
```

- [ ] **Step 4: Delete `models.py` and update `mononet/keras/__init__.py`**

```bash
git rm mononet/keras/models.py
```

```python
"""Keras 3 backend for mononet.

Uses `keras.ops`, so the same code runs whether the user has Keras set
to a JAX, TensorFlow, or PyTorch backend.
"""

from mononet.keras.layers import MonoDense, MonoInput, MonoResidual

__all__ = ["MonoDense", "MonoInput", "MonoResidual"]
```

- [ ] **Step 5: Add `_run_keras` to `tests/equivalence/test_mono_input.py`**

```python
def _run_keras(case: EquivalenceCase) -> np.ndarray:
    pytest.importorskip("keras")
    from keras import ops

    from mononet.core.types import MonotonicityMask
    from mononet.keras import MonoInput

    layer = MonoInput(MonotonicityMask(np.asarray(case.params["directions"], dtype=np.int8)))
    return np.asarray(layer(ops.convert_to_tensor(case.array("x"))))
```

and add `"keras": _run_keras` to `_RUNNERS`.

- [ ] **Step 6: Create `tests/keras/test_property_monotonic.py`**

```python
"""Property test: Keras layers are non-decreasing in every input."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from hypothesis import given, settings
from hypothesis import strategies as st
from keras import ops

from mononet.keras import MonoDense


@settings(deadline=None, max_examples=30)
@given(mode=st.sampled_from(["switch", "absolute"]), seed=st.integers(0, 10_000))
def test_mono_dense_nondecreasing(mode: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    layer = MonoDense(4, mode=mode)
    x = ops.convert_to_tensor(rng.normal(size=(6, 3)))
    bump = np.zeros((6, 3))
    bump[:, 1] = 0.5
    delta = layer(x + ops.convert_to_tensor(bump)) - layer(x)
    assert bool(ops.all(delta >= -1e-5))
```

- [ ] **Step 7: Run, expect pass**

Run: `MONONET_TEST_BACKEND=keras KERAS_BACKEND=jax uv run pytest tests/keras tests/equivalence -q` → PASS.

- [ ] **Step 8: Commit**

```bash
git add mononet/keras/layers.py mononet/keras/__init__.py tests/keras tests/equivalence/test_mono_input.py
git rm mononet/keras/models.py
git commit -m "feat(keras): MonoDense, MonoResidual, MonoInput layers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Phase 7 — Documentation, exports, and final verification

### Task 15: Update meta-spec + CLAUDE.md naming; mypy/ruff; full suite

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-05-21-mononet-package-design.md`
- Verify: full suite per backend, mypy, ruff.

**Interfaces:** none (docs + verification).

- [ ] **Step 1: Update `CLAUDE.md`** — in the "Multi-backend pattern" code block, change the `models.py` line to reflect there are no composed models, and update these lines:
  - Line ~57: `└── models.py  # public, composed models (MonoMLP, MonoFeatureBlock)` → remove the `models.py` row; note `layers.py` holds `MonoLinear/MonoDense`, `MonoResidual`, `MonoInput`.
  - Line ~62: replace the `models.py composes …` bullet with: "`layers.py` holds the layer and block classes; there are no composed model classes — users stack with the framework's native `Sequential`."
  - Line ~66: `MonoLinearConfig` → `MonoConfig`, `MonoResidualConfig`.
  - Lines ~70-73 (Naming): drop the `MonoMLP`/`MonoFeatureBlock` bullet; add `MonoResidual`/`MonoInput` share one name across backends; reference snake_case becomes `monotonic_dense`, `monotonic_residual`.

- [ ] **Step 2: Update `docs/superpowers/specs/2026-05-21-mononet-package-design.md`** — in the layout/naming/API sections (lines ~46-62, ~124-153, ~180-234): remove `models.py`, `MonoMLP`, `MonoFeatureBlock`, `monotonic_mlp`; rename `MonoLinearConfig` → `MonoConfig`; add `MonoResidual`/`MonoInput`; note `MonotonicityMask` is `{-1,+1}` and lives behind `MonoInput`. Add a one-line note that Sub-project A was redesigned 2026-06-27 (link the A spec).

- [ ] **Step 3: Update the mypy override in `pyproject.toml`**

`models.py` files are deleted, so drop the `mononet.*.models` entries from the `[[tool.mypy.overrides]]` `module` list, leaving only the three `*.layers` modules:

```toml
[[tool.mypy.overrides]]
module = [
    "mononet.torch.layers",
    "mononet.jax.layers",
    "mononet.keras.layers",
]
disallow_subclassing_any = false
```

- [ ] **Step 4: Verify the top-level import test still passes**

Run: `uv run pytest tests/test_top_level_imports.py -q` → PASS (core re-exports `MonotonicityMask`; no backend imported).

- [ ] **Step 5: Run mypy and ruff**

Run: `uv run ruff check --exit-non-zero-on-fix && uv run ruff format && uv run mypy`
Expected: clean. Fix any type/lint errors (common: add return annotations, `from __future__ import annotations`, narrow `Any`).

- [ ] **Step 6: Run the full suite per backend**

```bash
MONONET_TEST_BACKEND=torch uv run pytest -q
MONONET_TEST_BACKEND=jax   uv run pytest -q
MONONET_TEST_BACKEND=keras KERAS_BACKEND=jax uv run pytest -q
```
Expected: all PASS (uninstalled backends skip).

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md pyproject.toml docs/superpowers/specs/2026-05-21-mononet-package-design.md
git commit -m "docs: update meta-spec and CLAUDE.md for unified Mono layer naming

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Notes for the implementer

- **gelu must be the tanh approximation everywhere** (reference `np.tanh` formula, torch `approximate="tanh"`, jax `approximate=True`, keras `approximate=True`). The exact/erf variant will break equivalence tolerances.
- **Gradient tolerances** are looser (`atol=1e-4, rtol=1e-3`) than output tolerances because expected grads are central finite differences. Don't tighten them.
- **Switch `W⁻ = min(W, 0)`** (negative part kept negative) — `torch.clamp(max=0)`, `jnp.minimum(·,0)`, `ops.minimum(·,0)`. Never `max(0,-W)`.
- **Stateless kernels:** never mutate the stored `weight`; compute `|W|`, `W⁺`, `W⁻`, `exp(skip_weight)` on fresh tensors each forward pass.
- **MonoInput direction validation** comes free from `MonotonicityMask` (`{-1,+1}` only).
- If `MONONET_TEST_BACKEND` is unset it defaults to `torch` in the equivalence runners; CI sets it explicitly.
- **Object-or-literal (spec §3):** the code blocks implement resolution for `activation` (`str | ActivationSpec`), `init` (`str | InitSpec`), `F` (`Module | factory`), and `directions` (`int | MonotonicityMask`). Per the user's request, also accept an arbitrary **callable** for `activation` and the gates: in each layer's resolver add `if callable(x) and not isinstance(x, (str, ActivationSpec)): use it directly` and call the resolved function in the forward pass (bypassing the name/token dispatch in `_kernels`). Callables are the power-path and are deliberately **not** covered by the committed equivalence vectors (which use tokens only, since callables don't serialize through `MonoConfig`/`get_config`). Keep the token dispatch as the serializable default.
