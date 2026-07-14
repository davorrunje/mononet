# Phase 2: `alternate` construction + composition-aware `prev=` init — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `alternate` monotonic construction — per-layer pure activation alternating across depth (Sartor Prop 3.9) — with a composition-aware, pre-activation-centering initialization applied via an init-time `prev=` reference, across all three backends.

**Architecture:** `alternate` is an activation *layout* on the existing `|W|` map, resolved entirely in the layer: its forward reuses the `mixed` (`|W|`) kernel with a *pure* per-layer `convex_fraction ∈ {0,1}` (mathematically identical to `mixed` at `cf=0/1`, already covered by the `c0`/`c1` equivalence vectors), so **kernels, `reference.py`, and equivalence vectors are untouched**. The layer takes `prev=` (the preceding `alternate` layer), reads its phase (to alternate) and analytic output mean (to center this layer's pre-activation), computes `(gain, bias)` via a thin new core function, and discards the reference. `prev=None` ⇒ entry layer. Prototype-validated in `/tmp/prev_proto.py`.

**Tech Stack:** Python 3.11+, uv, pytest, ruff, mypy, pre-commit; PyTorch / JAX (Flax NNX) / Keras 3 via `MONONET_TEST_BACKEND`.

## Global Constraints

- Python 3.11+, line length 88 (ruff). Strict mypy. MyST field-list docstrings on public API.
- No Pydantic; stdlib dataclasses only. Init math stays pure NumPy in `mononet/core/init.py` (no backend import).
- Branch `feat/alternate-construction` (off `main`, already checked out). Commit `git commit --no-gpg-sign`. Never commit to `main`.
- `mode` valid values become exactly `"mixed"`, `"alternate"`, `"split"` (default `"mixed"`).
- **`alternate` never reaches the kernel/reference** — the layer translates its forward to kernel-mode `"mixed"`. The kernel/`reference.py` keep validating `∈ {"mixed","split"}` (unchanged); passing `"alternate"` to them raises, by design.
- `convex_fraction` is reserved for `mixed` — reject a non-default value under `mode="alternate"`. `prev=` is valid **only** under `mode="alternate"`.
- The `absolute_init_params` symbol name is kept (its rename is a separate deferred cleanup, not in this plan).
- Backends run here: torch, jax, keras (keras under `KERAS_BACKEND=jax`, set by the test harness).

## Design decisions (locked)

- **D1:** `Mode = Literal["mixed", "alternate", "split"]`; `MonoConfig`/`MonoResidualConfig` validation accepts all three. A flat config with `mode="alternate"` expresses intent; the composition-aware init is applied at build time via `prev=` (relevant to the Phase-4 benchmark builder, not to this plan).
- **D2:** Composition-aware init is a thin core function reusing `_solve_gain`/`_expect`. Since unit *output* variance fixes the pre-activation std to the per-activation gain `G = _solve_gain(name, 0.0)` regardless of `m_in`, the per-layer `gain = G / s` with `s = sqrt(1 + m_in²(1−2/π))`; `out_mean = ±E[act(G·H)]` is a per-activation constant (reached immediately, no transient).
- **D3:** The layer computes `weight_std = gain / sqrt(fan_in)` and `bias = −gain·sqrt(2/π)·sqrt(fan_in)·m_in` (pre-activation centering) — via a second core helper so the formula is defined and tested once.
- **D4:** `MonoResidual` (all three backends) **rejects `mode="alternate"`** with a `ValueError` pointing to a custom `F`. `alternate` needs the `prev=` chain, which the default-`F` builder does not thread — an unguarded `mode="alternate"` would silently build a stack of unchained all-convex entry layers (no alternation). Residual-alternation is a documented wash (spec §5.4), so this is a guard, not a feature.

---

### Task 1: Core init function + `Mode` literal + config validation

**Files:**
- Modify: `mononet/core/init.py` (add two functions after `absolute_init_params`, line 135; add `import math` or use `np`)
- Modify: `mononet/core/config.py` (line 16 `Mode`; validation lines 56 and 135)
- Test: `tests/core/test_alternating_init.py` (new), `tests/core/test_config.py`

**Interfaces:**
- Produces: `alternating_init_params(activation, m_in, convex) -> (gain, out_mean)` and `alternating_weight_bias(gain, m_in, fan_in) -> (weight_std, bias)` in `mononet.core.init`; `Mode = Literal["mixed","alternate","split"]`.

- [ ] **Step 1: Write failing core tests**

Create `tests/core/test_alternating_init.py`:
```python
import math

import numpy as np
import pytest

from mononet.core.init import (
    _expect,
    _solve_gain,
    alternating_init_params,
    alternating_weight_bias,
)


def test_entry_layer_is_unit_gain_zero_bias() -> None:
    # m_in=0 (entry): s=1 so gain == G (the mixed unit-variance gain); bias == 0.
    gain, out_mean = alternating_init_params("relu", m_in=0.0, convex=True)
    assert gain == pytest.approx(_solve_gain("relu", 0.0))
    ws, bias = alternating_weight_bias(gain, m_in=0.0, fan_in=32)
    assert bias == 0.0
    assert ws == pytest.approx(gain / math.sqrt(32))


def test_out_mean_is_signed_per_activation_constant() -> None:
    g_unit = _solve_gain("relu", 0.0)
    expected = _expect("relu", 0.0, g_unit, moment=1)  # E[relu(G·H)]
    conv_gain, conv_mean = alternating_init_params("relu", m_in=-0.4, convex=True)
    _, conc_mean = alternating_init_params("relu", m_in=0.4, convex=False)
    assert conv_mean == pytest.approx(expected)
    assert conc_mean == pytest.approx(-expected)


def test_interior_gain_shrinks_and_bias_centers() -> None:
    # interior layer fed by opposite class (m_in != 0): gain = G/s < G, bias sign
    # opposes m_in (pulls the |W|-inflated preactivation back to 0).
    g_unit = _solve_gain("elu", 0.0)
    m_in = 0.6
    gain, _ = alternating_init_params("elu", m_in=m_in, convex=False)
    s = math.sqrt(1.0 + m_in**2 * (1.0 - 2.0 / math.pi))
    assert gain == pytest.approx(g_unit / s)
    _, bias = alternating_weight_bias(gain, m_in=m_in, fan_in=16)
    assert bias < 0.0  # m_in > 0 -> negative centering bias


def test_montecarlo_output_variance_is_unit() -> None:
    # A pure layer built with these params has ~unit output variance under
    # a standard-normal-ish |W|·h + b, confirming the centering/gain.
    rng = np.random.default_rng(0)
    fan = 64
    m_in = 0.3
    gain, _ = alternating_init_params("relu", m_in=m_in, convex=True)
    ws, bias = alternating_weight_bias(gain, m_in, fan)
    W = np.abs(rng.normal(0.0, ws, size=(fan, 256)))  # |W|
    h = rng.normal(m_in, 1.0, size=(4096, fan))       # input mean m_in, unit var
    z = h @ W + bias
    out = np.maximum(0.0, z)  # relu (convex)
    assert out.var() == pytest.approx(1.0, abs=0.15)


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        alternating_init_params("gelu", m_in=0.0, convex=True)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/core/test_alternating_init.py -v`
Expected: FAIL (ImportError: `alternating_init_params` not defined).

- [ ] **Step 3: Implement the two core functions**

In `mononet/core/init.py`, add `import math` near the top (after `from __future__`), then after line 135 append:
```python
def alternating_init_params(
    activation: ActivationSpec | str, m_in: float, convex: bool
) -> tuple[float, float]:
    """Derive ``(gain, out_mean)`` for one layer of the ``alternate`` construction.

    Composition-aware, pre-activation-centering init for a *pure* (all-convex or
    all-concave) ``|W|`` layer whose input has per-coordinate mean ``m_in`` and unit
    variance. Reuses the ``mixed`` unit-variance gain ``G`` (:func:`_solve_gain`):
    forcing unit *output* variance fixes the pre-activation std to ``G`` regardless of
    ``m_in``, so ``gain = G / s`` with ``s = sqrt(1 + m_in**2 * (1 - 2/pi))``. The
    per-coordinate output mean is a per-activation constant ``E[act(G*H)]``, signed by
    the layer's class; feed it back as the next layer's ``m_in``.

    :param activation: Base activation name or :class:`ActivationSpec`.
    :param m_in: Per-coordinate mean of the layer input (``0.0`` for the entry layer).
    :param convex: Whether this layer uses the convex activation (else its concave
        reflection).
    :returns: ``(gain, out_mean)``.
    :raises ValueError: If the activation is unknown.
    """
    name = activation if isinstance(activation, str) else activation.name
    _act(name, np.zeros(1))  # validate name early (raises on unknown)
    g_unit = _solve_gain(name, 0.0)
    out_convex = _expect(name, 0.0, g_unit, moment=1)
    s = math.sqrt(1.0 + m_in * m_in * (1.0 - 2.0 / math.pi))
    gain = g_unit / s
    return gain, (out_convex if convex else -out_convex)


def alternating_weight_bias(
    gain: float, m_in: float, fan_in: int
) -> tuple[float, float]:
    """Weight std and pre-activation-centering bias for an ``alternate`` layer.

    :param gain: Per-layer gain from :func:`alternating_init_params`.
    :param m_in: Per-coordinate mean of the layer input.
    :param fan_in: Number of input features.
    :returns: ``(weight_std, bias)`` — ``weight_std = gain / sqrt(fan_in)``,
        ``bias = -gain * sqrt(2/pi) * sqrt(fan_in) * m_in``.
    """
    root_fan = math.sqrt(fan_in)
    return gain / root_fan, -gain * math.sqrt(2.0 / math.pi) * root_fan * m_in
```

- [ ] **Step 4: Run core init tests green**

Run: `uv run pytest tests/core/test_alternating_init.py -v`
Expected: PASS.

- [ ] **Step 5: Widen `Mode` + config validation (TDD)**

Add to `tests/core/test_config.py`:
```python
def test_alternate_mode_accepted() -> None:
    from mononet.core.config import MonoConfig

    cfg = MonoConfig(units=4, mode="alternate")
    assert cfg.mode == "alternate"
    assert cfg.to_dict()["mode"] == "alternate"
    assert MonoConfig.from_json(cfg.to_json()).mode == "alternate"
```
Run it → FAIL (validation rejects "alternate"). Then in `mononet/core/config.py`:
- Line 16: `Mode = Literal["mixed", "alternate", "split"]`
- Line 56: `if self.mode not in ("mixed", "alternate", "split"):`
- Line 57: `raise ValueError(f"mode must be 'mixed', 'alternate', or 'split'; got {self.mode!r}")`
- Same edits at MonoResidualConfig lines 135-136.

- [ ] **Step 6: Run core tests green**

Run: `uv run pytest tests/core -v`
Expected: PASS (existing hard-break tests still pass — old names still rejected).

- [ ] **Step 7: Commit**

```bash
git add mononet/core tests/core
git commit --no-gpg-sign -m "feat(core): composition-aware alternate init + add 'alternate' to Mode"
```

---

### Task 2: torch `MonoLinear` — `prev=` + `alternate`

**Files:**
- Modify: `mononet/torch/layers.py` (`MonoLinear.__init__` 77-111, `forward` 113-122; docstring 58-75)
- Test: `tests/torch/test_alternate.py` (new)

**Interfaces:**
- Consumes: `alternating_init_params`, `alternating_weight_bias` (Task 1).
- Produces: `MonoLinear(..., mode="alternate", prev=<MonoLinear|None>)`; instance attrs `_alt_convex: bool`, `_alt_out_mean: float`; forward resolves `alternate`→kernel-mode `"mixed"`.

- [ ] **Step 1: Write failing torch tests**

Create `tests/torch/test_alternate.py`:
```python
import numpy as np
import pytest
import torch
from torch import nn

from mononet.torch import MonoLinear


def _stack(act="relu", depth=4, d=4, h=16):
    torch.manual_seed(0)
    layers, prev, prev_in = [], None, d
    for _ in range(depth):
        lay = MonoLinear(prev_in, h, mode="alternate", activation=act, prev=prev)
        layers.append(lay)
        prev, prev_in = lay, h
    layers.append(MonoLinear(prev_in, 1, mode="mixed", activation="identity"))
    return nn.Sequential(*layers)


def test_prev_alternates_phase_and_entry_is_convex() -> None:
    net = _stack()
    alt = [m for m in net if getattr(m, "mode", None) == "alternate"]
    assert [m._alt_convex for m in alt] == [True, False, True, False]


def test_entry_bias_zero_interior_bias_alternates_sign() -> None:
    net = _stack()
    alt = [m for m in net if m.mode == "alternate"]
    assert alt[0].bias.detach().abs().max().item() == pytest.approx(0.0, abs=1e-6)
    assert alt[1].bias.detach().mean().item() < 0.0  # concave interior
    assert alt[2].bias.detach().mean().item() > 0.0  # convex interior


def test_prev_not_retained() -> None:
    net = _stack()
    alt = [m for m in net if m.mode == "alternate"]
    assert all("prev" not in vars(m) for m in alt)
    assert not any("prev" in k for k in net.state_dict())


def test_alternate_is_monotone_nondecreasing() -> None:
    net = _stack()
    x = torch.zeros(1, 4)
    with torch.no_grad():
        base = net(x)
        for j in range(4):
            bumped = x.clone()
            bumped[0, j] += 1e-2
            assert (net(bumped) - base).item() >= -1e-5


def test_convex_fraction_rejected_for_alternate() -> None:
    with pytest.raises(ValueError, match="convex_fraction"):
        MonoLinear(4, 8, mode="alternate", activation="relu", convex_fraction=0.3)


def test_prev_rejected_for_non_alternate() -> None:
    entry = MonoLinear(4, 8, mode="alternate", activation="relu")
    with pytest.raises(ValueError, match="prev"):
        MonoLinear(8, 8, mode="mixed", activation="relu", prev=entry)


def test_prev_must_be_alternate() -> None:
    mixed = MonoLinear(4, 8, mode="mixed", activation="relu")
    with pytest.raises(ValueError, match="alternate"):
        MonoLinear(8, 8, mode="alternate", activation="relu", prev=mixed)


def test_deep_alternate_trains_stably() -> None:
    # depth-8 plain alternate stack does not diverge (contrast: mixed diverges).
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.uniform(-1, 1, (2000, 4)), dtype=torch.float32)
    y = torch.tensor(
        (1 / (1 + np.exp(-3 * (x.numpy() - 0.1)))).sum(1, keepdims=True),
        dtype=torch.float32,
    )
    y = (y - y.mean()) / y.std()
    net = _stack(depth=8)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    lf = nn.MSELoss()
    for _ in range(300):
        opt.zero_grad()
        lf(net(x), y).backward()
        opt.step()
    with torch.no_grad():
        assert lf(net(x), y).item() < 0.9  # beats predict-the-mean (~1.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_alternate.py -v`
Expected: FAIL (`MonoLinear` has no `prev` kwarg / `mode="alternate"` mis-inits).

- [ ] **Step 3: Implement**

Add the import at top of `mononet/torch/layers.py`:
```python
from mononet.core.init import (
    absolute_init_params,
    alternating_init_params,
    alternating_weight_bias,
)
```
(merge with the existing `absolute_init_params` import.)

Add `prev` to the signature (after `near_zero_scale`):
```python
        near_zero_scale: float | None = None,
        prev: MonoLinear | None = None,
```
Replace the init body (lines 91-106) with:
```python
        self.mode = mode
        self.activation_name = (
            "identity" if activation is None else _act_name(activation)
        )
        self.weight = nn.Parameter(torch.empty(in_features, units))
        bias_fill = 0.0
        if mode == "alternate":
            if convex_fraction != 0.5:
                raise ValueError(
                    "convex_fraction is not configurable for mode='alternate'"
                )
            if init is not None:
                raise ValueError("init is not configurable for mode='alternate'")
            convex, m_in = self._alternate_phase(prev)
            self.convex_fraction = 1.0 if convex else 0.0
            gain, out_mean = alternating_init_params(
                self.activation_name, m_in, convex
            )
            w_std, bias_fill = alternating_weight_bias(gain, m_in, in_features)
            self._alt_convex = convex
            self._alt_out_mean = out_mean
            with torch.no_grad():
                self.weight.normal_(0.0, w_std)
        else:
            if prev is not None:
                raise ValueError("prev is only valid for mode='alternate'")
            self.convex_fraction = convex_fraction
            if mode == "mixed" and init is None:
                gain, bias_fill = absolute_init_params(
                    self.activation_name, convex_fraction
                )
                with torch.no_grad():
                    self.weight.normal_(0.0, gain / math.sqrt(in_features))
            else:
                _init_weight(self.weight, init)
        self.bias = nn.Parameter(torch.full((units,), bias_fill)) if bias else None
```
Add the phase helper as a static method on `MonoLinear`:
```python
    @staticmethod
    def _alternate_phase(prev: MonoLinear | None) -> tuple[bool, float]:
        """Return ``(convex, m_in)`` for an alternate layer given its predecessor."""
        if prev is None:
            return True, 0.0  # entry: convex, standardized input
        if getattr(prev, "mode", None) != "alternate":
            raise ValueError("prev must be an alternate-mode MonoLinear")
        return (not prev._alt_convex), prev._alt_out_mean
```
In `forward`, resolve the kernel mode:
```python
        kernel_mode = "mixed" if self.mode == "alternate" else self.mode
        return _kernels.monotonic_dense(
            x, self.weight, bias, kernel_mode, self.activation_name, self.convex_fraction
        )
```
Update the class docstring: document `mode="alternate"`, the `prev` param, and that `convex_fraction`/`init` are not configurable in alternate mode.

Add the **MonoResidual guard (D4)** — at the top of `MonoResidual.__init__` (torch, after `super().__init__()`):
```python
        if mode == "alternate":
            raise ValueError(
                "mode='alternate' is not supported in MonoResidual; build a custom "
                "F of alternate MonoLinear layers chained with prev= instead"
            )
```
Add a test to `tests/torch/test_alternate.py`:
```python
def test_mono_residual_rejects_alternate() -> None:
    from mononet.torch import MonoResidual

    with pytest.raises(ValueError, match="alternate"):
        MonoResidual(8, 8, mode="alternate", activation="relu")
```

- [ ] **Step 4: Run torch tests green**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch -v`
Expected: PASS (new alternate tests + all existing torch tests).

- [ ] **Step 5: Commit**

```bash
git add mononet/torch tests/torch
git commit --no-gpg-sign -m "feat(torch): alternate mode via prev= (composition-aware init)"
```

---

### Task 3: jax `MonoLinear` — `prev=` + `alternate`

**Files:**
- Modify: `mononet/jax/layers.py` (`MonoLinear.__init__` 85-121, `__call__` 123-141; docstring)
- Test: `tests/jax/test_alternate.py` (new)

**Interfaces:** Same public surface as Task 2, Flax NNX idioms (`nnx.Param`, `rngs`). `prev: MonoLinear | None = None` before the `rngs` keyword-only arg.

- [ ] **Step 1: Write failing jax tests**

Create `tests/jax/test_alternate.py` mirroring `tests/torch/test_alternate.py`, adapted to NNX: build with `rngs=nnx.Rngs(0)`, pass `rngs=` to each `MonoLinear`, read `m._alt_convex`, `m.bias[...]`, and finite-difference via `jnp`. For "prev not retained", assert `"prev" not in vars(m)` and that `nnx.state(net)` contains no `prev` leaf. Use the same reject-case tests.

- [ ] **Step 2: Run to verify failure**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax/test_alternate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Import `alternating_init_params, alternating_weight_bias` alongside `absolute_init_params`. Add `prev: MonoLinear | None = None` before `rngs`. Mirror Task 2's branch structure, but initialise the weight through NNX:
```python
        if mode == "alternate":
            if convex_fraction != 0.5:
                raise ValueError(
                    "convex_fraction is not configurable for mode='alternate'"
                )
            if init is not None:
                raise ValueError("init is not configurable for mode='alternate'")
            convex, m_in = self._alternate_phase(prev)
            self.convex_fraction = 1.0 if convex else 0.0
            gain, out_mean = alternating_init_params(
                self.activation_name, m_in, convex
            )
            w_std, bias_fill = alternating_weight_bias(gain, m_in, in_features)
            self._alt_convex = convex
            self._alt_out_mean = out_mean
            w = jinit.normal(stddev=w_std)(rngs.params(), (in_features, units))
            self.weight = nnx.Param(w)
        else:
            if prev is not None:
                raise ValueError("prev is only valid for mode='alternate'")
            self.convex_fraction = convex_fraction
            ... (existing mixed/split branch, unchanged) ...
```
Add the same `_alternate_phase` static method. In `__call__`, use `kernel_mode = "mixed" if self.mode == "alternate" else self.mode`. Store `_alt_convex`/`_alt_out_mean` as plain Python attributes (NOT `nnx.Param`/`nnx.Variable`) so NNX does not treat them as state — confirm `nnx.state(net)` has no `_alt_*`/`prev` leaves in the test. Add the **MonoResidual guard (D4)**: raise `ValueError` on `mode="alternate"` at the top of the jax `MonoResidual.__init__` (same message as torch) + a mirror test.

- [ ] **Step 4: Run jax tests green**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mononet/jax tests/jax
git commit --no-gpg-sign -m "feat(jax): alternate mode via prev= (composition-aware init)"
```

---

### Task 4: keras `MonoDense` — `prev=` + `alternate` (build-time weights)

**Files:**
- Modify: `mononet/keras/layers.py` (`MonoDense.__init__` 72-95, `build` 97-129, `call` 131-145, `get_config` 147-163; docstring)
- Test: `tests/keras/test_alternate.py` (new)

**Interfaces:** `MonoDense(units, ..., mode="alternate", prev=<MonoDense|None>)`. Because keras builds lazily, split the work: the **mean chain** (`convex`, `m_in`, `gain`, `out_mean`) is fan-in-independent and resolved in `__init__`; the fan-in-dependent `(weight_std, bias)` is computed in `build`.

- [ ] **Step 1: Write failing keras tests**

Create `tests/keras/test_alternate.py` mirroring the torch tests (harness sets `KERAS_BACKEND=jax`). Build a `keras.Sequential` of `MonoDense(..., mode="alternate", prev=...)`; call the model once on a dummy batch to trigger `build`; then assert phases via `m._alt_convex`, entry/interior bias via `m.b`, monotonicity via finite difference, `prev` not retained (`"prev" not in vars(m)`), and the reject cases. Include the deep-stack stability smoke.

- [ ] **Step 2: Run to verify failure**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_alternate.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Import `alternating_init_params, alternating_weight_bias`. Add `prev` param. In `__init__`, after the existing field assignments:
```python
        self._is_alternate = mode == "alternate"
        if self._is_alternate:
            if convex_fraction != 0.5:
                raise ValueError(
                    "convex_fraction is not configurable for mode='alternate'"
                )
            if init is not None:
                raise ValueError("init is not configurable for mode='alternate'")
            convex, self._alt_m_in = self._alternate_phase(prev)
            self.convex_fraction = 1.0 if convex else 0.0
            self._alt_gain, out_mean = alternating_init_params(
                self.activation_name, self._alt_m_in, convex
            )
            self._alt_convex = convex
            self._alt_out_mean = out_mean
        elif prev is not None:
            raise ValueError("prev is only valid for mode='alternate'")
```
Add the same `_alternate_phase` static method (returns `(convex, m_in)`; validates `prev._is_alternate`). In `build`, branch before the existing `_absolute_default` path:
```python
        if self._is_alternate:
            w_std, bias_fill = alternating_weight_bias(
                self._alt_gain, self._alt_m_in, in_f
            )
            w_init = keras.initializers.RandomNormal(stddev=w_std)
            b_init = keras.initializers.Constant(bias_fill)
        elif self._absolute_default:
            ... (existing) ...
        else:
            ... (existing) ...
```
In `call`, `kernel_mode = "mixed" if self.mode == "alternate" else self.mode`. In `get_config`, keep emitting `mode` (already does) — `prev` is an init-time reference, NOT serialized (document that a deserialized alternate layer loses its `prev` chain; re-chaining is a build concern — acceptable, note in docstring). Add the **MonoResidual guard (D4)**: raise `ValueError` on `mode="alternate"` at the top of the keras `MonoResidual.__init__` (same message as torch) + a mirror test.

- [ ] **Step 4: Run keras tests green**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mononet/keras tests/keras
git commit --no-gpg-sign -m "feat(keras): alternate mode via prev= (build-time composition-aware init)"
```

---

### Task 5: Full-repo verification + PR

**Files:** none (verification only).

- [ ] **Step 1: Full suite, all backends**

```bash
MONONET_TEST_BACKEND=torch uv run pytest
MONONET_TEST_BACKEND=jax   uv run pytest
MONONET_TEST_BACKEND=keras uv run pytest
```
Expected: PASS. Confirm equivalence still 24/24 on each (alternate does not touch the kernels, so vectors are unaffected).

- [ ] **Step 2: Lint, types, hooks**

```bash
uv run ruff check --exit-non-zero-on-fix
uv run ruff format --check
uv run mypy
uv run pre-commit run --all-files
```
Expected: all PASS (`reference-hash` unaffected — `reference.py` unchanged; docs build green).

- [ ] **Step 3: Cross-backend init sanity (one command)**

Confirm the three backends derive the same per-layer `(gain, bias)` for an identical alternate chain (they all call the same NumPy core):
```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_alternate.py::test_entry_bias_zero_interior_bias_alternates_sign -q
MONONET_TEST_BACKEND=jax   uv run pytest tests/jax/test_alternate.py -q -k bias
MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_alternate.py -q -k bias
```
Expected: PASS on all three.

- [ ] **Step 4: Push + PR**

```bash
git push -u origin feat/alternate-construction
gh pr create --base main --title "feat: alternate monotonic construction + composition-aware prev= init" \
  --body-file - <<'EOF'
Phase 2 of the monotone-constructions spec: the `alternate` construction
(per-layer pure activation alternating across depth) + composition-aware,
pre-activation-centering initialization applied via an init-time `prev=`
reference. Layer-resolved to the `|W|` kernel path — kernels, `reference.py`,
and equivalence vectors are untouched. All three backends; full green.

Unblocks the Phase-4 flavor ablation.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
```

---

## Notes for later phases

- **Phase 3** distills the construction/init/residual writeup into `docs/concepts/`.
- **Phase 4** wires the benchmark harness (`_ALL_FLAVORS`, `model_builder` `prev=` chaining, `alt_init` selector) per `docs/superpowers/specs/2026-07-14-flavor-ablation-benchmark-design.md`.
- Deferred cleanups (separate): rename `absolute_init_params`→`mixed_init_params` + `_absolute_default`; rename equivalence case filename slugs; rename `test_absolute_init.py`.
