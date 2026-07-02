# Correct Static Init for `absolute` (Deep-Net Trainability) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `mode="absolute"` a static, data-free, per-activation initialization (variance-preserving `gain` + layer-mean-centering `bias`) so deep `absolute` stacks train, across torch/JAX/Keras — with a committed diagnostic, a fast regression test, and a deep synthetic benchmark exported to docs.

**Architecture:** One backend-agnostic NumPy helper `mononet/core/init.py:absolute_init_params(activation, convex_fraction) -> (gain, bias)` derives the init from the activation's moments under `N(0,1)` (Gauss–Hermite quadrature). Each backend's layer applies it as the default for `mode="absolute"`. Diagnostics + deep benchmark live in `benchmarks/`; results render in `docs/benchmarks/deep-init.ipynb`.

**Tech Stack:** Python 3.11, NumPy (core helper), PyTorch/JAX(Flax NNX)/Keras 3 (layer wiring + per-backend tests), pytest, Sphinx + myst-nb.

## Global Constraints

- Branch **`feat/absolute-init`** (already created off main); never commit to `main`.
- Commit **UNSIGNED** during subagent execution (`git -c commit.gpgsign=false commit`); controller re-signs the whole branch before push.
- `mononet/core/init.py` is **NumPy-only** — no torch/jax/keras import (preserves lazy backend imports; `import mononet` must not import a backend).
- Run mypy as **`uv run --group bench mypy`**. Per-task gates: `uv run ruff check`, `uv run --group bench mypy`, the task's pytest. Final: `uv run pre-commit run --all-files --hook-stage manual` + `./tools/build-docs.sh`.
- No Pydantic; stdlib dataclasses. MyST field-list docstrings (`:param:`/`:returns:`/`:raises:`, no `:type:`) on public functions. ruff line-length 88; strict mypy; type hints on every function.
- `benchmarks/` never ships in the wheel; no `[project.scripts]`. Result JSON written with a trailing newline; never commit `*.db`/`*.jsonl`.
- The fix is **the default when `mode="absolute"` and `init is None`**; an explicit `InitSpec`/str still overrides the weight init (bias then stays zero). `switch` default init is unchanged. Kernels + equivalence harness untouched.
- `b = 0` at the default `convex_fraction = 0.5` for all activations; the gain is the whole default fix.

---

### Task 1: Core init helper `absolute_init_params`

**Files:**
- Create: `mononet/core/init.py`
- Test: `tests/core/test_init.py`

**Interfaces:**
- Consumes: `mononet.core.types.ActivationSpec` (for the name; the helper accepts `ActivationSpec | str`).
- Produces: `absolute_init_params(activation: ActivationSpec | str, convex_fraction: float) -> tuple[float, float]` returning `(gain, bias)`. `gain > 0`; `bias == 0.0` when `convex_fraction == 0.5`. Raises `ValueError` on unknown activation.

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_init.py`:

```python
import numpy as np
import pytest

from mononet.core.init import absolute_init_params, _act, _expect  # helpers used below

ACTS = ["relu", "elu", "selu", "softplus"]


@pytest.mark.parametrize("act", ACTS)
def test_gain_preserves_output_variance(act: str) -> None:
    gain, bias = absolute_init_params(act, 0.5)
    # pre-activation ~ N(bias, gain^2); output variance must be ~1
    var = _expect(act, bias, gain, moment=2) - _expect(act, bias, gain, moment=1) ** 2
    assert gain > 0.0
    assert abs(var - 1.0) < 1e-2


@pytest.mark.parametrize("act", ACTS)
def test_bias_zero_at_half(act: str) -> None:
    _, bias = absolute_init_params(act, 0.5)
    assert bias == 0.0


@pytest.mark.parametrize("act", ACTS)
@pytest.mark.parametrize("f", [0.25, 0.75])
def test_layer_mean_zero_off_half(act: str, f: float) -> None:
    gain, bias = absolute_init_params(act, f)
    # layer mean = f*E[act(H+b)] - (1-f)*E[act(-(H+b))], H~N(0,1)
    conv = _expect(act, bias, gain, moment=1)
    conc = _expect(act, -bias, gain, moment=1)  # E[act(-H - b)] via H symmetry
    layer_mean = f * conv - (1.0 - f) * conc
    assert abs(layer_mean) < 1e-2


def test_deterministic() -> None:
    assert absolute_init_params("elu", 0.5) == absolute_init_params("elu", 0.5)


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        absolute_init_params("gelu", 0.5)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/test_init.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'mononet.core.init'`.

- [ ] **Step 3: Implement `mononet/core/init.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Static, data-free initialization derivation for the ``absolute`` construction.

Pure NumPy (no backend import). Derives, from an activation's moments under a
standard normal pre-activation, a variance-preserving weight ``gain`` and a
layer-mean-centering ``bias`` for ``mode="absolute"``.
"""

from __future__ import annotations

import numpy as np

from mononet.core.types import ActivationSpec

_GH_DEG = 64  # Gauss-Hermite nodes for E_{H~N(0,1)}[.]
# probabilists' Gauss-Hermite: E[f(H)] = sum(_W * f(_X)), H ~ N(0, 1)
_X, _W_RAW = np.polynomial.hermite_e.hermegauss(_GH_DEG)
_W = _W_RAW / np.sqrt(2.0 * np.pi)


def _act(name: str, h: np.ndarray) -> np.ndarray:
    """NumPy mirror of the backend base activations.

    :param name: One of ``relu``, ``elu``, ``selu``, ``softplus``.
    :param h: Pre-activation values.
    :returns: Activated values, same shape as ``h``.
    :raises ValueError: If ``name`` is not a known activation.
    """
    if name == "relu":
        return np.maximum(0.0, h)
    if name == "elu":
        return np.where(h > 0.0, h, np.expm1(h))
    if name == "selu":
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * np.where(h > 0.0, h, alpha * np.expm1(h))
    if name == "softplus":
        return np.logaddexp(0.0, h)
    raise ValueError(f"unknown activation {name!r}")


def _expect(name: str, mean: float, scale: float, *, moment: int) -> float:
    """``E_{H~N(0,1)}[ act(scale*H + mean)^moment ]`` via Gauss-Hermite.

    :param name: Activation name.
    :param mean: Added to the scaled node (the pre-activation mean).
    :param scale: Multiplies the node (the pre-activation std).
    :param moment: 1 for the mean, 2 for the second moment.
    :returns: The expectation as a float.
    """
    vals = _act(name, scale * _X + mean)
    return float(np.sum(_W * vals**moment))


def _variance(name: str, mean: float, scale: float) -> float:
    return _expect(name, mean, scale, moment=2) - _expect(name, mean, scale, moment=1) ** 2


def _bisect(f, lo: float, hi: float, *, tol: float = 1e-9, iters: int = 200) -> float:
    """Bisection root of a monotone ``f`` on ``[lo, hi]`` (``f(lo)``/``f(hi)`` bracket 0)."""
    flo = f(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if (fmid > 0.0) == (flo > 0.0):
            lo, flo = mid, fmid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _solve_gain(name: str, bias: float) -> float:
    """Gain s.t. ``Var[act(gain*H + bias)] = 1`` (variance is increasing in gain)."""
    return _bisect(lambda g: _variance(name, bias, g) - 1.0, 1e-4, 20.0)


def _solve_bias(name: str, gain: float, f: float) -> float:
    """Scalar bias s.t. the layer mean ``f*E[act(H+b)] - (1-f)*E[act(-(H+b))] = 0``.

    Monotone increasing in ``b`` (both terms shift the layer mean up with ``b``).
    """

    def layer_mean(b: float) -> float:
        conv = _expect(name, b, gain, moment=1)
        conc = _expect(name, -b, gain, moment=1)  # E[act(-H - b)] via H symmetry
        return f * conv - (1.0 - f) * conc

    return _bisect(layer_mean, -20.0, 20.0)


def absolute_init_params(
    activation: ActivationSpec | str, convex_fraction: float
) -> tuple[float, float]:
    """Derive ``(gain, bias)`` for the ``absolute`` construction.

    The weight init std is ``gain / sqrt(fan_in)`` (variance-preserving through the
    ``|W|`` + convex/concave-activation map), and the whole bias vector is
    initialised to the scalar ``bias`` (which centers a layer's output mean). At
    ``convex_fraction == 0.5`` the split is self-cancelling so ``bias == 0`` and the
    default fix is purely the gain. Both are data-free (Gauss-Hermite quadrature),
    so the init stays static and seed-reproducible.

    :param activation: Base activation name or :class:`ActivationSpec`.
    :param convex_fraction: Fraction of convex units in the layer.
    :returns: ``(gain, bias)`` — ``gain > 0``; ``bias == 0.0`` when ``convex_fraction == 0.5``.
    :raises ValueError: If the activation is unknown.
    """
    name = activation if isinstance(activation, str) else activation.name
    _act(name, np.zeros(1))  # validate name early (raises on unknown)
    if convex_fraction == 0.5:
        return _solve_gain(name, 0.0), 0.0
    gain = _solve_gain(name, 0.0)
    bias = 0.0
    for _ in range(8):  # fixed-point: gain and bias couple mildly off f=0.5
        bias = _solve_bias(name, gain, convex_fraction)
        gain = _solve_gain(name, bias)
    return gain, bias
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/core/test_init.py -q`
Expected: PASS (all parametrizations).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/core/init.py tests/core/test_init.py && uv run --group bench mypy`
Expected: clean; `Success: no issues found`. (If mypy flags the untyped `f` param of `_bisect`, annotate it `f: Callable[[float], float]` with a `TYPE_CHECKING` import of `Callable`.)

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/core/init.py tests/core/test_init.py
git -c commit.gpgsign=false commit -m "feat(core): absolute_init_params — variance gain + layer-mean bias"
```

---

### Task 2: Torch wiring — default `absolute` init

**Files:**
- Modify: `mononet/torch/layers.py` (`MonoLinear.__init__`)
- Test: `tests/torch/test_absolute_init.py`

**Interfaces:**
- Consumes: `absolute_init_params(activation, convex_fraction) -> (gain, bias)` (Task 1).
- Produces: `MonoLinear(..., mode="absolute")` with `init is None` initialises `weight ~ N(0, gain²/fan_in)` and `bias` filled with the scalar `bias`. Explicit `init` overrides weight (bias stays 0). `switch` unchanged.

- [ ] **Step 1: Write the failing test**

Create `tests/torch/test_absolute_init.py`:

```python
import math

import pytest

torch = pytest.importorskip("torch")

from mononet.core.init import absolute_init_params
from mononet.torch import MonoLinear


def test_absolute_default_weight_scale_and_bias() -> None:
    torch.manual_seed(0)
    in_f, units = 256, 512
    layer = MonoLinear(in_f, units, mode="absolute", activation="elu")
    gain, bias = absolute_init_params("elu", 0.5)
    got = float(layer.weight.detach().std())
    assert abs(got - gain / math.sqrt(in_f)) < 0.05 * gain / math.sqrt(in_f)
    assert torch.allclose(
        layer.bias.detach(), torch.full((units,), bias, dtype=layer.bias.dtype)
    )


def test_absolute_bias_nonzero_off_half() -> None:
    torch.manual_seed(0)
    layer = MonoLinear(64, 64, mode="absolute", activation="elu", convex_fraction=0.25)
    _, bias = absolute_init_params("elu", 0.25)
    assert bias != 0.0
    assert torch.allclose(
        layer.bias.detach(), torch.full((64,), bias, dtype=layer.bias.dtype)
    )


def test_explicit_init_overrides_absolute() -> None:
    torch.manual_seed(0)
    layer = MonoLinear(64, 64, mode="absolute", activation="elu", init="he_normal")
    assert torch.allclose(layer.bias.detach(), torch.zeros(64, dtype=layer.bias.dtype))


def test_switch_default_unchanged() -> None:
    torch.manual_seed(0)
    layer = MonoLinear(64, 64, mode="switch", activation="elu")
    assert torch.allclose(layer.bias.detach(), torch.zeros(64, dtype=layer.bias.dtype))
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_absolute_init.py -q`
Expected: FAIL — default `absolute` currently uses `he_normal` and zero bias, so the weight-scale and non-zero-bias assertions fail.

- [ ] **Step 3: Implement in `mononet/torch/layers.py`**

Add `import math` at the top (after `from __future__ import annotations`), and import the helper:

```python
from mononet.core.init import absolute_init_params
```

Replace the weight/bias construction in `MonoLinear.__init__` (currently):

```python
        self.weight = nn.Parameter(torch.empty(in_features, units))
        _init_weight(self.weight, init)
        self.bias = nn.Parameter(torch.zeros(units)) if bias else None
```

with:

```python
        self.weight = nn.Parameter(torch.empty(in_features, units))
        bias_fill = 0.0
        if mode == "absolute" and init is None:
            gain, bias_fill = absolute_init_params(self.activation_name, convex_fraction)
            with torch.no_grad():
                self.weight.normal_(0.0, gain / math.sqrt(in_features))
        else:
            _init_weight(self.weight, init)
        self.bias = (
            nn.Parameter(torch.full((units,), bias_fill)) if bias else None
        )
```

(`self.activation_name` and `convex_fraction` are already set earlier in `__init__`.)

- [ ] **Step 4: Run to verify it passes**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_absolute_init.py tests/torch/test_public_api.py -q`
Expected: PASS (new file green; existing torch API tests still green).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/torch/layers.py tests/torch/test_absolute_init.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/torch/layers.py tests/torch/test_absolute_init.py
git -c commit.gpgsign=false commit -m "feat(torch): default absolute init uses absolute_init_params"
```

---

### Task 3: JAX wiring — default `absolute` init

**Files:**
- Modify: `mononet/jax/layers.py` (`MonoLinear.__init__`)
- Test: `tests/jax/test_absolute_init.py`

**Interfaces:**
- Consumes: `absolute_init_params` (Task 1).
- Produces: JAX `MonoLinear(..., mode="absolute")` with `init is None` → `weight ~ N(0, gain²/fan_in)`, `bias` filled with scalar `bias`.

- [ ] **Step 1: Write the failing test**

Create `tests/jax/test_absolute_init.py`:

```python
import math

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from mononet.core.init import absolute_init_params
from mononet.jax import MonoLinear


def test_absolute_default_weight_scale_and_bias() -> None:
    in_f, units = 256, 512
    layer = MonoLinear(in_f, units, mode="absolute", activation="elu", rngs=nnx.Rngs(0))
    gain, bias = absolute_init_params("elu", 0.5)
    got = float(jnp.std(layer.weight[...]))
    assert abs(got - gain / math.sqrt(in_f)) < 0.05 * gain / math.sqrt(in_f)
    assert jnp.allclose(layer.bias[...], jnp.full((units,), bias))


def test_absolute_bias_nonzero_off_half() -> None:
    layer = MonoLinear(64, 64, mode="absolute", activation="elu",
                       convex_fraction=0.25, rngs=nnx.Rngs(0))
    _, bias = absolute_init_params("elu", 0.25)
    assert bias != 0.0
    assert jnp.allclose(layer.bias[...], jnp.full((64,), bias))


def test_switch_default_unchanged() -> None:
    layer = MonoLinear(64, 64, mode="switch", activation="elu", rngs=nnx.Rngs(0))
    assert jnp.allclose(layer.bias[...], jnp.zeros((64,)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax/test_absolute_init.py -q`
Expected: FAIL — default `absolute` weight uses `he_normal`, bias is zeros.

- [ ] **Step 3: Implement in `mononet/jax/layers.py`**

Add imports near the top:

```python
import math

import jax.nn.initializers as jinit  # already imported
from mononet.core.init import absolute_init_params
```

Replace the weight/bias construction in `MonoLinear.__init__` (currently):

```python
        self.weight = nnx.Param(_init_array((in_features, units), init, rngs))
        self.bias: nnx.Param[jnp.ndarray] | None = (
            nnx.Param(jnp.zeros((units,))) if bias else None
        )
```

with:

```python
        bias_fill = 0.0
        if mode == "absolute" and init is None:
            gain, bias_fill = absolute_init_params(self.activation_name, convex_fraction)
            w = jinit.normal(stddev=gain / math.sqrt(in_features))(
                rngs.params(), (in_features, units)
            )
            self.weight = nnx.Param(w)
        else:
            self.weight = nnx.Param(_init_array((in_features, units), init, rngs))
        self.bias: nnx.Param[jnp.ndarray] | None = (
            nnx.Param(jnp.full((units,), bias_fill)) if bias else None
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax/test_absolute_init.py tests/jax -q`
Expected: PASS (new file + existing JAX tests).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/jax/layers.py tests/jax/test_absolute_init.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/jax/layers.py tests/jax/test_absolute_init.py
git -c commit.gpgsign=false commit -m "feat(jax): default absolute init uses absolute_init_params"
```

---

### Task 4: Keras wiring — default `absolute` init

**Files:**
- Modify: `mononet/keras/layers.py` (`MonoDense.__init__`, `MonoDense.build`)
- Test: `tests/keras/test_absolute_init.py`

**Interfaces:**
- Consumes: `absolute_init_params` (Task 1).
- Produces: Keras `MonoDense(..., mode="absolute")` with `init is None` → kernel `RandomNormal(stddev=gain/√fan_in)`, bias `Constant(bias)`.

- [ ] **Step 1: Write the failing test**

Create `tests/keras/test_absolute_init.py`:

```python
import math

import numpy as np
import pytest

pytest.importorskip("keras")

from mononet.core.init import absolute_init_params
from mononet.keras import MonoDense


def _build(units: int, in_f: int, **kw) -> MonoDense:
    layer = MonoDense(units, **kw)
    layer.build((None, in_f))
    return layer


def test_absolute_default_weight_scale_and_bias() -> None:
    in_f, units = 256, 512
    layer = _build(units, in_f, mode="absolute", activation="elu")
    gain, bias = absolute_init_params("elu", 0.5)
    got = float(np.std(np.asarray(layer.w)))
    assert abs(got - gain / math.sqrt(in_f)) < 0.05 * gain / math.sqrt(in_f)
    assert np.allclose(np.asarray(layer.b), np.full((units,), bias))


def test_absolute_bias_nonzero_off_half() -> None:
    layer = _build(64, 64, mode="absolute", activation="elu", convex_fraction=0.25)
    _, bias = absolute_init_params("elu", 0.25)
    assert bias != 0.0
    assert np.allclose(np.asarray(layer.b), np.full((64,), bias))


def test_switch_default_unchanged() -> None:
    layer = _build(64, 64, mode="switch", activation="elu")
    assert np.allclose(np.asarray(layer.b), np.zeros((64,)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_absolute_init.py -q`
Expected: FAIL — default `absolute` kernel uses `he_normal`, bias `zeros`.

- [ ] **Step 3: Implement in `mononet/keras/layers.py`**

Add imports:

```python
import math

from mononet.core.init import absolute_init_params
```

In `MonoDense.__init__`, keep `self.init_name = _init_name(init)` and also record whether the default derivation applies:

```python
        self.init_name = _init_name(init)
        self._absolute_default = mode == "absolute" and init is None
```

In `MonoDense.build`, replace the weight/bias creation:

```python
        self.w = self.add_weight(
            shape=(int(input_shape[-1]), self.units),
            initializer=self.init_name,
            trainable=True,
            name="weight",
        )
        self.b = (
            self.add_weight(
                shape=(self.units,),
                initializer="zeros",
                trainable=True,
                name="bias",
            )
            if self.use_bias
            else None
        )
```

with:

```python
        in_f = int(input_shape[-1])
        if self._absolute_default:
            gain, bias_fill = absolute_init_params(self.activation_name, self.convex_fraction)
            w_init = keras.initializers.RandomNormal(stddev=gain / math.sqrt(in_f))
            b_init = keras.initializers.Constant(bias_fill)
        else:
            w_init = self.init_name
            b_init = "zeros"
        self.w = self.add_weight(
            shape=(in_f, self.units), initializer=w_init, trainable=True, name="weight"
        )
        self.b = (
            self.add_weight(
                shape=(self.units,), initializer=b_init, trainable=True, name="bias"
            )
            if self.use_bias
            else None
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_absolute_init.py tests/keras -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/keras/layers.py tests/keras/test_absolute_init.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/keras/layers.py tests/keras/test_absolute_init.py
git -c commit.gpgsign=false commit -m "feat(keras): default absolute init uses absolute_init_params"
```

---

### Task 5: Diagnostic engine (`benchmarks/_common/init_diagnostics.py`)

**Files:**
- Create: `benchmarks/_common/init_diagnostics.py`
- Test: `tests/benchmarks/test_init_diagnostics.py`

**Interfaces:**
- Consumes: `mononet.torch.MonoLinear`.
- Produces:
  - `synthetic_monotone(n: int, d: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]` — standardized `X` (n×d), target `y = Σ softplus(aᵢ·xᵢ) + ε` with `aᵢ>0`, `y` standardized.
  - `grad_flow(mode: str, depth: int, *, activation: str = "elu", width: int = 32, seed: int = 0) -> dict[str, float | list[float]]` — `{"input_grad_norm": float, "layer_grad_norms": list[float]}`.
  - `trainability(mode: str, depth: int, *, activation: str = "elu", epochs: int = 100, seed: int = 0) -> dict[str, float]` — `{"final_train_loss": float, "epochs_to_threshold": float}`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_init_diagnostics.py`:

```python
import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.init_diagnostics import (
    grad_flow,
    synthetic_monotone,
    trainability,
)


def test_synthetic_monotone_shapes_and_standardized() -> None:
    X, y = synthetic_monotone(256, 5, seed=0)
    assert X.shape == (256, 5)
    assert y.shape == (256,)
    assert abs(float(X.mean())) < 0.1 and abs(float(X.std()) - 1.0) < 0.1


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_grad_flow_finite(mode: str) -> None:
    out = grad_flow(mode, depth=4, activation="elu", width=16, seed=0)
    assert np.isfinite(out["input_grad_norm"])
    assert len(out["layer_grad_norms"]) == 4
    assert all(np.isfinite(g) for g in out["layer_grad_norms"])


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_trainability_finite(mode: str) -> None:
    out = trainability(mode, depth=2, activation="elu", epochs=3, seed=0)
    assert np.isfinite(out["final_train_loss"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_init_diagnostics.py -q`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `benchmarks/_common/init_diagnostics.py`**

```python
"""Diagnostics for absolute-vs-switch init conditioning across depth (torch)."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mononet.torch import MonoLinear


def synthetic_monotone(n: int, d: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Standardized features with a known monotone target.

    :param n: Number of samples.
    :param d: Number of features.
    :param seed: RNG seed.
    :returns: ``(X, y)`` with ``X`` standardized ``(n, d)`` and ``y`` standardized ``(n,)``;
        ``y = Σ softplus(aᵢ·xᵢ) + ε``, ``aᵢ > 0`` (non-decreasing in every feature).
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    a = rng.uniform(0.5, 1.5, size=d)
    y = np.logaddexp(0.0, x * a).sum(axis=1) + 0.05 * rng.standard_normal(n)
    y = (y - y.mean()) / (y.std() + 1e-8)
    return x, y


def _stack(mode: str, depth: int, d: int, width: int, activation: str) -> nn.Module:
    layers: list[nn.Module] = [
        MonoLinear(d, width, mode=mode, activation=activation)
    ]
    layers += [
        MonoLinear(width, width, mode=mode, activation=activation)
        for _ in range(depth - 1)
    ]
    layers.append(MonoLinear(width, 1, mode=mode, activation=activation))
    return nn.Sequential(*[layer.double() for layer in layers])


def grad_flow(
    mode: str, depth: int, *, activation: str = "elu", width: int = 32, seed: int = 0
) -> dict[str, float | list[float]]:
    """Init-time gradient flow through an untrained plain stack.

    :param mode: ``switch`` or ``absolute``.
    :param depth: Number of hidden ``MonoLinear`` layers.
    :param activation: Base activation.
    :param width: Hidden width.
    :param seed: RNG seed.
    :returns: ``{"input_grad_norm": float, "layer_grad_norms": [float, ...]}`` (len == depth).
    """
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    net = _stack(mode, depth, x_np.shape[1], width, activation)
    x = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    loss = nn.functional.mse_loss(net(x), y)
    loss.backward()
    hidden = [m for m in net if isinstance(m, MonoLinear)][:depth]
    return {
        "input_grad_norm": float(x.grad.norm()),
        "layer_grad_norms": [float(m.weight.grad.norm()) for m in hidden],
    }


def trainability(
    mode: str, depth: int, *, activation: str = "elu", epochs: int = 100, seed: int = 0
) -> dict[str, float]:
    """Fixed-budget train loss of a plain stack on the synthetic target.

    :param mode: ``switch`` or ``absolute``.
    :param depth: Number of hidden ``MonoLinear`` layers.
    :param activation: Base activation.
    :param epochs: Full-batch training epochs.
    :param seed: RNG seed.
    :returns: ``{"final_train_loss": float, "epochs_to_threshold": float}`` (threshold 0.5 MSE;
        ``inf`` if never reached).
    """
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    net = _stack(mode, depth, x_np.shape[1], 32, activation)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    hit = float("inf")
    loss_val = float("inf")
    for ep in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()
        opt.step()
        loss_val = float(loss)
        if loss_val < 0.5 and hit == float("inf"):
            hit = float(ep)
    return {"final_train_loss": loss_val, "epochs_to_threshold": hit}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/benchmarks/test_init_diagnostics.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check benchmarks/_common/init_diagnostics.py tests/benchmarks/test_init_diagnostics.py && uv run --group bench mypy`
Expected: clean. (If mypy flags `x.grad` / `m.weight.grad` as `Tensor | None`, guard with `assert x.grad is not None`.)

- [ ] **Step 6: Commit (unsigned)**

```bash
git add benchmarks/_common/init_diagnostics.py tests/benchmarks/test_init_diagnostics.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): init diagnostics (synthetic monotone, grad-flow, trainability)"
```

---

### Task 6: Fast regression test — deep `absolute` gradient band

**Files:**
- Create: `tests/torch/test_deep_init.py`
- Create: `tests/jax/test_deep_init.py`
- Create: `tests/keras/test_deep_init.py`

**Interfaces:**
- Consumes: backend `MonoLinear`/`MonoDense` at the new default `absolute` init.

- [ ] **Step 1: Write the failing test (torch)**

Create `tests/torch/test_deep_init.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear


def test_deep_absolute_gradient_band() -> None:
    # A depth-8 absolute stack at the default init must have a well-conditioned
    # input-gradient at init: neither vanishing nor exploding.
    torch.manual_seed(0)
    width = 32
    layers = [MonoLinear(8, width, mode="absolute", activation="elu")]
    layers += [MonoLinear(width, width, mode="absolute", activation="elu") for _ in range(7)]
    layers.append(MonoLinear(width, 1, mode="absolute", activation="elu"))
    net = torch.nn.Sequential(*[layer.double() for layer in layers])
    x = torch.randn(256, 8, dtype=torch.float64, requires_grad=True)
    net(x).sum().backward()
    g = float(x.grad.norm() / x.shape[0] ** 0.5)  # per-sample input-grad norm
    assert 1e-2 < g < 1e2, f"input-grad norm {g} out of band (vanish/explode)"
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_deep_init.py -q`
Expected: **This must be run AFTER Task 2.** With the fix it passes; to see it fail-first, temporarily confirm the band would be violated by the old `he_normal` default (documented, not committed). If Task 2 is already merged, the test passes directly — acceptable, as this is a regression guard, not a red-first unit for new code.

- [ ] **Step 3: Add JAX + Keras equivalents**

`tests/jax/test_deep_init.py`:

```python
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax
import jax.numpy as jnp
from flax import nnx

from mononet.jax import MonoLinear


def test_deep_absolute_gradient_band() -> None:
    width = 32
    rngs = nnx.Rngs(0)
    layers = [MonoLinear(8, width, mode="absolute", activation="elu", rngs=rngs)]
    layers += [
        MonoLinear(width, width, mode="absolute", activation="elu", rngs=rngs)
        for _ in range(7)
    ]
    layers.append(MonoLinear(width, 1, mode="absolute", activation="elu", rngs=rngs))

    def fwd(x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for layer in layers:
            h = layer(h)
        return h.sum()

    x = jax.random.normal(jax.random.key(0), (256, 8))
    g = jax.grad(fwd)(x)
    norm = float(jnp.linalg.norm(g) / x.shape[0] ** 0.5)
    assert 1e-2 < norm < 1e2, f"input-grad norm {norm} out of band"
```

`tests/keras/test_deep_init.py`:

```python
import numpy as np
import pytest

pytest.importorskip("keras")

import keras
from keras import ops

from mononet.keras import MonoDense


def test_deep_absolute_gradient_band() -> None:
    width = 32
    layers = [MonoDense(width, mode="absolute", activation="elu") for _ in range(8)]
    layers.append(MonoDense(1, mode="absolute", activation="elu"))
    x = ops.convert_to_tensor(np.random.default_rng(0).standard_normal((256, 8)))
    import tensorflow as tf  # keras default backend in CI is jax; use keras grad path

    # Backend-agnostic gradient via keras: use a GradientTape-like through ops is
    # backend-specific; instead assert forward is finite and well-scaled as a proxy.
    h = x
    for layer in layers:
        h = layer(h)
    out = float(ops.convert_to_numpy(ops.std(h)))
    assert np.isfinite(out) and out < 1e3, f"deep absolute output std {out} exploded"
```

Note: Keras 3's grad API is backend-specific; the Keras test uses a forward-conditioning proxy (output std bounded) rather than an input-gradient norm, keeping it backend-agnostic and CI-fast. Torch/JAX assert the true gradient band.

- [ ] **Step 4: Run to verify they pass**

Run each under its backend:
```
MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_deep_init.py -q
MONONET_TEST_BACKEND=jax   uv run pytest tests/jax/test_deep_init.py -q
MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_deep_init.py -q
```
Expected: PASS on each active backend.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check tests/torch/test_deep_init.py tests/jax/test_deep_init.py tests/keras/test_deep_init.py && uv run --group bench mypy`
Expected: clean. (Remove the unused `tensorflow`/`tf` import in the Keras test if ruff flags F401 — the proxy path does not need it.)

- [ ] **Step 6: Commit (unsigned)**

```bash
git add tests/torch/test_deep_init.py tests/jax/test_deep_init.py tests/keras/test_deep_init.py
git -c commit.gpgsign=false commit -m "test: deep absolute init gradient/conditioning band (all backends)"
```

---

### Task 7: Docs notebook + toctree + README

**Files:**
- Create: `docs/benchmarks/deep-init.ipynb`
- Modify: `docs/benchmarks/index.md` (Sections bullet + hidden toctree)
- Modify: `benchmarks/README.md` (a short "Deep-network init" subsection)
- Create: `benchmarks/results/deep-init/.gitignore` (`*.db`, `*.jsonl`)

**Interfaces:**
- Consumes: committed JSON under `benchmarks/results/deep-init/` (written by Task 8); `benchmarks._common.init_diagnostics`.

- [ ] **Step 1: Create `benchmarks/results/deep-init/.gitignore`**

```
*.db
*.jsonl
```

- [ ] **Step 2: Create `docs/benchmarks/deep-init.ipynb`**

A rendered notebook with a **missing-results guard** so the docs build is green before Task 8's maintainer run. Cell 1 (markdown): title + one paragraph explaining the `absolute` init problem and the fix (link `protocol.md` is unrelated — link the spec instead). Cell 2 (code):

```python
import json
from pathlib import Path

import pandas as pd

RESULTS = Path("../../benchmarks/results/deep-init")
files = sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []
if not files:
    print("No deep-init results committed yet. Run the maintainer sweep "
          "(see benchmarks/README.md, 'Deep-network init').")
else:
    rows = [json.loads(f.read_text()) for f in files]
    df = pd.DataFrame(rows)
    # depth-sweep table: mode × depth × {final_train_loss, input_grad_norm}
    display(df.pivot_table(index="depth", columns="mode",
                           values=["final_train_loss", "input_grad_norm"]))
```

- [ ] **Step 3: Wire into `docs/benchmarks/index.md`**

Add a Sections bullet (after the Protocol bullet):

```markdown
- [Deep-network init](deep-init.ipynb) — why deep `absolute` stacks need a tailored
  initialization, and evidence that the fix trains them.
```

Add `deep-init` to the hidden toctree (after `protocol`):

```markdown
```{toctree}
:hidden:
:maxdepth: 2

protocol
deep-init
00-overview
paper-reproduction/index
flavor-comparison
```
```

- [ ] **Step 4: README subsection**

Append to `benchmarks/README.md`:

```markdown
## Deep-network init

Deep `absolute` stacks require a tailored static init (`mononet.core.init.absolute_init_params`,
now the default for `mode="absolute"`). To reproduce the diagnostic + deep-net showcase:

```bash
uv run --extra torch --group bench python -m benchmarks.deep_init_run   # writes results/deep-init/*.json
uv run --group bench --group docs --extra torch jupyter nbconvert --to notebook \
  --execute --inplace docs/benchmarks/deep-init.ipynb
```
See `docs/benchmarks/deep-init.ipynb` for the rendered curves.
```

- [ ] **Step 5: Verify docs build + pre-commit**

Run: `uv run pre-commit run --all-files --hook-stage manual && ./tools/build-docs.sh`
Expected: pre-commit clean (codespell/EOF — notebooks may be re-fixed once; re-run until clean); Sphinx build succeeds with `deep-init` in the toctree and the missing-results guard printing the placeholder.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add docs/benchmarks/deep-init.ipynb docs/benchmarks/index.md benchmarks/README.md benchmarks/results/deep-init/.gitignore
git -c commit.gpgsign=false commit -m "docs(benchmarks): deep-init notebook + toctree + README (missing-results guard)"
```

---

### Task 8: Controller phase — calibrate, run, render, ship

> **Not a subagent TDD task.** The controller runs this after Tasks 1–7 pass review and the whole-branch review is clean. It performs the D sweep (validates the derived constants), the deep-net benchmark, renders the notebook, re-signs, and opens the PR.

- [ ] **Step 1: Validate the derived init via the D sweep**

Run a script over `benchmarks._common.init_diagnostics` for `mode ∈ {switch,absolute}`, `activation ∈ {elu,relu}`, `depth ∈ {1,2,4,8,16,32}`: record `grad_flow` + `trainability`, and per-layer forward mean/variance. Confirm the decision rule from the spec: post-fix, `absolute` tracks `switch` (per-layer mean ≈ 0, variance ratio ≈ 1; train loss no longer plateaus with depth). If the gain/bias are off, adjust the quadrature degree or the solve bounds in `mononet/core/init.py` (Task 1) and re-review that task.

- [ ] **Step 2: Create `benchmarks/deep_init_run.py`** (thin runner)

A small module writing `benchmarks/results/deep-init/<mode>-<activation>-d<depth>.json` (each with `depth`, `mode`, `activation`, `final_train_loss`, `input_grad_norm`, per-layer stats), trailing newline. Train a genuinely deep `absolute` net (depth ~16–32) on `synthetic_monotone` and record the curve. Commit the runner + the JSON.

- [ ] **Step 3: Re-render the notebook + commit results**

```bash
uv run --extra torch --group bench python -m benchmarks.deep_init_run
uv run --group bench --group docs --extra torch jupyter nbconvert --to notebook \
  --execute --inplace docs/benchmarks/deep-init.ipynb
git add benchmarks/deep_init_run.py benchmarks/results/deep-init/*.json docs/benchmarks/deep-init.ipynb
git -c commit.gpgsign=false commit -m "bench(deep-init): committed sweep + deep-net results, rendered notebook"
```

- [ ] **Step 4: Re-sign, push, PR**

```bash
git rebase --exec "git commit --amend --no-edit -n -S" origin/main
git log --format="%h %G? %s" origin/main..HEAD   # expect all G
git push -u origin feat/absolute-init
gh pr create --title "Correct static init for absolute (deep-net trainability)" --body-file <(...)
```

- [ ] **Step 5: Confirm CI green**

```bash
gh pr checks <n>
```
Expected: all 15 test legs + static-analysis + pre-commit + docs-smoke pass.

---

## Notes for the executor

- Run mypy as `uv run --group bench mypy` (the canonical CI gate); a bare `uv run mypy` lacks typer/optuna and false-fails on `benchmarks/`.
- Backend tests must be run with the matching `MONONET_TEST_BACKEND` and use `pytest.importorskip`.
- `mononet/core/init.py` must stay NumPy-only — importing it must not pull in torch/jax/keras (protects `import mononet` laziness). A quick guard: `python -c "import sys, mononet.core.init; assert 'torch' not in sys.modules"`.
- Tasks 2–4 are independent given Task 1; they may be reviewed in any order but each rebased on the latest branch tip.
- The Task-6 gradient-band constants (`1e-2 < g < 1e2`) are deliberately wide (a vanish/explode guard, not a tight fit); Task 8 may tighten them from the measured post-fix D values and note it.
