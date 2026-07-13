# MonoResidual Gate Instability Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make deep `MonoResidual` stacks actually use their depth by fixing two independent traps — the `scaled_elu` gate dead zone (→ `softplus` gate) and the `|W|` exact-zero gradient fixed point (→ near-zero init of `F`'s last layer) — across all three backends, with reproducible evidence and evidence-backed docs.

**Architecture:** Two orthogonal changes shipped as the new `MonoResidual` default. (B) A new `softplus` gate token added to the NumPy reference and the torch/jax/keras kernels, made the default `beta_gate`. (A) Near-zero initialization (`weight *= 1e-3`, `bias = 0`) of the default `F` sub-module's **last** layer, applied at each backend's weight-creation site via a private `near_zero_out` flag on the dense layer. The stateless kernels are unchanged except for the new gate branch; near-zero init is a layer-construction concern only.

**Tech Stack:** Python 3.11+, PyTorch, JAX (Flax NNX), Keras 3, NumPy reference, Optuna (benchmarks), pytest, uv, ruff, mypy (strict), Sphinx + myst-nb.

## Global Constraints

- Python 3.11+, line length 88 (ruff). Strict mypy; type hints on every function/method.
- MyST field-list docstrings (`:param:`, `:returns:`, `:raises:`) on all public functions/classes; types from signatures only.
- Stdlib `dataclasses` only — **no Pydantic**.
- Lazy backend imports: never import torch/jax/keras from the top-level `mononet/__init__.py`. Backend tests use `pytest.importorskip`.
- Monotonicity is a hard invariant: `g_α, g_β ≥ 0` for all parameter values; `F` non-decreasing. No signed/ReZero gate. Every change must preserve the monotonicity property tests.
- `_NEAR_ZERO_SCALE = 1e-3` (the near-identity warm-start scale; stable band ≈`1e-3`, `≥1e-2` re-blows-up). Define once per backend `layers.py`.
- The package was never publicly released — change defaults outright, no compat shims. Keep `scaled_elu` as a selectable token (used by the ablation), just not the default.
- Commit proactively on this branch (`fix/monoresidual-instability`); never commit to `main`.
- Backend equivalence: run the equivalence suite with `MONONET_TEST_BACKEND=torch` locally (the `default` devcontainer has torch); jax/keras are verified in CI.
- Spec: [`docs/superpowers/specs/2026-07-13-monoresidual-gate-instability-fix-design.md`](../specs/2026-07-13-monoresidual-gate-instability-fix-design.md). Read it before starting.

---

### Task 1: `softplus` gate token in the NumPy reference

**Files:**
- Modify: `mononet/core/reference.py` (`apply_gate`, ~lines 60-75)
- Test: `tests/core/test_reference_activations.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `apply_gate("softplus", raw)` returns `np.logaddexp(0.0, raw)` — a strictly-positive gate, value `ln 2 ≈ 0.6931` at `raw=0`, gradient `σ(raw) ∈ (0,1)` everywhere (nonzero for `raw < 0`).

- [ ] **Step 1: Write the failing test**

Add to `tests/core/test_reference_activations.py`:

```python
def test_softplus_gate_value_and_gradient() -> None:
    zero = np.array(0.0)
    # value at 0 is ln 2 (unlike scaled_elu's eps and shifted_elu's 1)
    assert ref.apply_gate("softplus", zero) == pytest.approx(np.log(2.0))
    # strictly positive everywhere, including well into the negative side
    x = np.linspace(-20.0, 20.0, 200)
    assert np.all(ref.apply_gate("softplus", x) > 0.0)
    # gradient is nonzero on the negative side (no dead zone): sigmoid(-5) > 0
    h = 1e-6
    neg = -5.0
    d = (
        ref.apply_gate("softplus", np.array(neg + h))
        - ref.apply_gate("softplus", np.array(neg - h))
    ) / (2 * h)
    assert d == pytest.approx(1.0 / (1.0 + np.exp(-neg)), abs=1e-4)
    assert d > 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/core/test_reference_activations.py::test_softplus_gate_value_and_gradient -v`
Expected: FAIL with `ValueError: unknown gate token 'softplus'`.

- [ ] **Step 3: Add the softplus branch to `apply_gate`**

In `mononet/core/reference.py`, inside `apply_gate`, add before the final `raise`:

```python
    if token == "softplus":
        return np.logaddexp(0.0, raw)  # type: ignore[no-any-return]
```

Also update the docstring `:param token:` line to mention `softplus` (value `ln 2`, gradient `sigmoid`, no dead zone).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/core/test_reference_activations.py -v`
Expected: PASS (all gate tests).

- [ ] **Step 5: Commit**

```bash
git add mononet/core/reference.py tests/core/test_reference_activations.py
git commit -m "feat(core): add softplus gate token to the NumPy reference"
```

---

### Task 2: `softplus` gate token in the three backend kernels + equivalence cases

**Files:**
- Modify: `mononet/torch/_kernels.py` (`gate`, ~lines 52-68)
- Modify: `mononet/jax/_kernels.py` (`gate`, ~lines 46-56)
- Modify: `mononet/keras/_kernels.py` (`gate`, ~lines 46-56)
- Modify: `tools/regenerate-cases.py` (`_residual_cases`, ~lines 151-227)
- Regenerate: `tests/equivalence/cases/mono_residual/*.json`, `tests/equivalence/cases/REFERENCE_HASH`

**Interfaces:**
- Consumes: `apply_gate("softplus", …)` semantics from Task 1.
- Produces: `_kernels.gate("softplus", raw)` in each backend, numerically equal to the reference within the committed tolerances. New committed equivalence cases with `beta_gate="softplus"`.

- [ ] **Step 1: Add the softplus branch to each backend kernel**

`mononet/torch/_kernels.py`, in `gate`, before the final `raise`:

```python
    if token == "softplus":
        return functional.softplus(raw)
```

`mononet/jax/_kernels.py`, in `gate`, before the final `raise`:

```python
    if token == "softplus":
        return jnn.softplus(raw)
```

`mononet/keras/_kernels.py`, in `gate`, before the final `raise`:

```python
    if token == "softplus":
        return ops.softplus(raw)
```

Update each `gate` docstring `:param token:` to list `softplus`.

- [ ] **Step 2: Extend the equivalence case generator with softplus cases**

In `tools/regenerate-cases.py`, `_residual_cases`, add a `beta_gate` element to each grid tuple and two new softplus rows, and thread `beta_gate` through `fwd` and `params`:

```python
def _residual_cases() -> None:
    grid = [
        ("4x3x3-identity-switch", 4, 3, 3, None, "switch", "relu", "scaled_elu"),
        ("4x2x5-proj-switch", 4, 2, 5, (2, 5), "switch", "relu", "scaled_elu"),
        ("6x4x4-identity-abs", 6, 4, 4, None, "absolute", "elu", "scaled_elu"),
        ("5x3x3-identity-abs-id", 5, 3, 3, None, "absolute", "identity", "scaled_elu"),
        ("6x4x4-identity-abs-softplus", 6, 4, 4, None, "absolute", "elu", "softplus"),
        ("4x2x5-proj-switch-softplus", 4, 2, 5, (2, 5), "switch", "relu", "softplus"),
    ]
    for name, b, n, m, proj, mode, act, beta_gate in grid:
        rng = np.random.default_rng(_seed(name))
        x = rng.normal(size=(b, n))
        w = rng.normal(size=(n, m))
        bias = rng.normal(size=m)
        alpha = np.array(0.3)
        beta = np.array(0.5)
        skip = rng.normal(size=(n, m)) if proj else None
        spec = ActivationSpec(act)  # type: ignore[arg-type]

        def fwd(
            *,
            w: np.ndarray = w,  # type: ignore[type-arg]
            bias: np.ndarray = bias,  # type: ignore[type-arg]
            alpha: np.ndarray = alpha,  # type: ignore[type-arg]
            beta: np.ndarray = beta,  # type: ignore[type-arg]
            skip: np.ndarray | None = skip,  # type: ignore[type-arg]
            _x: np.ndarray = x,  # type: ignore[type-arg]
            _mode: str = mode,
            _spec: ActivationSpec = spec,
            _beta_gate: str = beta_gate,
        ) -> np.ndarray:  # type: ignore[type-arg]
            return ref.monotonic_residual(
                _x,
                w,
                bias,
                alpha,
                beta,
                mode=_mode,
                activation=_spec,
                beta_gate=_beta_gate,
                skip_weight=skip,
            )

        out = fwd()
        grads: dict[str, Any] = {
            "weights": _fd_grad(lambda v: fwd(w=v), w).tolist(),
            "bias": _fd_grad(lambda v: fwd(bias=v), bias).tolist(),
            "alpha": _fd_grad(lambda v: fwd(alpha=v), alpha).tolist(),
            "beta": _fd_grad(lambda v: fwd(beta=v), beta).tolist(),
        }
        inputs: dict[str, Any] = {
            "x": x.tolist(),
            "weights": w.tolist(),
            "bias": bias.tolist(),
            "alpha": alpha.tolist(),
            "beta": beta.tolist(),
        }
        if skip is not None:
            inputs["skip_weight"] = skip.tolist()
            grads["skip_weight"] = _fd_grad(lambda v: fwd(skip=v), skip).tolist()
        _write(
            "mono_residual",
            name,
            {
                "name": name,
                "inputs": inputs,
                "params": {
                    "mode": mode,
                    "activation": act,
                    "convex_fraction": 0.5,
                    "alpha_gate": "shifted_elu",
                    "beta_gate": beta_gate,
                    "has_projection": skip is not None,
                    "dtype": "float64",
                },
                "expected_output": out.tolist(),
                "expected_grads": grads,
                "atol": OUT_ATOL,
                "rtol": OUT_RTOL,
            },
        )
```

- [ ] **Step 3: Regenerate the committed cases**

Run: `uv run python tools/regenerate-cases.py`
Expected: writes the 6 `mono_residual` case files (2 new softplus ones) and a fresh `REFERENCE_HASH` (reference.py changed in Task 1). Prints `Written REFERENCE_HASH: <sha>`.

- [ ] **Step 4: Run the equivalence suite (torch) to verify cross-backend parity**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/equivalence/test_mono_residual.py -v`
Expected: PASS, including the two `*-softplus` cases (torch kernel matches the reference expected outputs/grads).

- [ ] **Step 5: Commit**

```bash
git add mononet/torch/_kernels.py mononet/jax/_kernels.py mononet/keras/_kernels.py \
        tools/regenerate-cases.py tests/equivalence/cases/mono_residual tests/equivalence/cases/REFERENCE_HASH
git commit -m "feat(kernels): add softplus gate token to all backends + equivalence cases"
```

---

### Task 3: Make `softplus` the default `beta_gate`

**Files:**
- Modify: `mononet/core/config.py` (`MonoResidualConfig.beta_gate`, ~line 112)
- Modify: `mononet/core/reference.py` (`monotonic_residual` signature default, ~line 131)
- Modify: `mononet/torch/_kernels.py` (~line 117), `mononet/jax/_kernels.py` (~line 105), `mononet/keras/_kernels.py` (~line 107) — `monotonic_residual` default `beta_gate`
- Modify: `mononet/torch/layers.py` (~line 145), `mononet/jax/layers.py` (~line 164), `mononet/keras/layers.py` (~line 182) — `MonoResidual.__init__` default `beta_gate`
- Test: `tests/core/test_config.py` (~line 43), `tests/core/test_reference_residual.py`

**Interfaces:**
- Consumes: the `softplus` token from Tasks 1-2.
- Produces: every default-constructed `MonoResidual`/`MonoResidualConfig`/`monotonic_residual` uses `beta_gate="softplus"`; `alpha_gate` stays `"shifted_elu"`.

- [ ] **Step 1: Update the config default assertion (failing test)**

In `tests/core/test_config.py::test_mono_residual_config_roundtrip`, change:

```python
    assert cfg.beta_gate == "softplus"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/core/test_config.py::test_mono_residual_config_roundtrip -v`
Expected: FAIL (`assert 'scaled_elu' == 'softplus'`).

- [ ] **Step 3: Flip the default everywhere**

Change `beta_gate: str = "scaled_elu"` → `beta_gate: str = "softplus"` in each of:
- `mononet/core/config.py` (`MonoResidualConfig`)
- `mononet/core/reference.py` (`monotonic_residual`)
- `mononet/torch/_kernels.py`, `mononet/jax/_kernels.py`, `mononet/keras/_kernels.py` (`monotonic_residual`)
- `mononet/torch/layers.py`, `mononet/jax/layers.py`, `mononet/keras/layers.py` (`MonoResidual.__init__`)

Update the corresponding `:param beta_gate:` docstrings that name the default `scaled_elu`.

- [ ] **Step 4: Update the reference residual default test**

`tests/core/test_reference_residual.py` has a test asserting the default block is `≈ identity` because `beta gate ≈ 0` (comment at line 26). With `softplus`, `g_β(0)=ln2≈0.69`, so that assumption no longer holds for the default. Update that test to pass an explicit `beta_gate="scaled_elu"` where it relies on `g_β ≈ 0`:

```python
    # near-identity holds only for the eps-gate; pin it explicitly here
    y = ref.monotonic_residual(x, w, b, alpha, beta, activation=spec, beta_gate="scaled_elu")
```

(Read the test first; apply the `beta_gate="scaled_elu"` argument to the specific `monotonic_residual` call whose assertion is `y ≈ identity skip`.)

- [ ] **Step 5: Run the affected tests**

Run: `uv run pytest tests/core/test_config.py tests/core/test_reference_residual.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mononet/core/config.py mononet/core/reference.py \
        mononet/torch/_kernels.py mononet/jax/_kernels.py mononet/keras/_kernels.py \
        mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py \
        tests/core/test_config.py tests/core/test_reference_residual.py
git commit -m "feat: default MonoResidual beta_gate to softplus (fixes gate dead-zone trap)"
```

---

### Task 4: Near-zero init of `F`'s last layer (all three backends)

**Files:**
- Modify: `mononet/torch/layers.py` (`MonoLinear.__init__`, `MonoResidual` default-`F` builder)
- Modify: `mononet/jax/layers.py` (`MonoLinear.__init__`, `MonoResidual` default-`F` builder)
- Modify: `mononet/keras/layers.py` (`MonoDense.__init__`/`build`, `MonoResidual` default-`F` builder)
- Test: `tests/torch/test_mono_residual_gate.py` (NEW), `tests/jax/test_mono_residual_gate.py` (NEW), `tests/keras/test_mono_residual_gate.py` (NEW)

**Interfaces:**
- Consumes: nothing new.
- Produces: a private `near_zero_out: bool = False` keyword on `MonoLinear` (torch/jax) and `MonoDense` (keras). When `True`, the layer's weight is initialized then multiplied by `_NEAR_ZERO_SCALE = 1e-3` and its bias set to zero. `MonoResidual`'s default `F` builds its **last** sub-layer with `near_zero_out=True` (both the `k==1` single-layer and the `k>1` last-of-stack cases). A custom `F` is untouched.

- [ ] **Step 1: Write the failing test (torch)**

Create `tests/torch/test_mono_residual_gate.py`:

```python
"""Near-zero init of the default F, and gate defaults, for MonoResidual (torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear, MonoResidual  # noqa: E402


def _last_linear(block: MonoResidual) -> MonoLinear:
    f = block.F
    if isinstance(f, MonoLinear):
        return f
    return f[-1]  # nn.Sequential


def test_default_F_last_layer_is_near_zero_but_nonzero() -> None:
    torch.manual_seed(0)
    block = MonoResidual(32, 32, mode="absolute", activation="elu")
    last = _last_linear(block)
    wnorm = float(last.weight.detach().abs().sum())
    # small but NOT exactly zero (exact zero would freeze under |W|)
    assert wnorm > 0.0
    assert wnorm < 1.0  # heavily attenuated vs a normal init (~tens)
    # bias zeroed
    assert last.bias is not None
    assert float(last.bias.detach().abs().sum()) == 0.0


def test_default_block_is_near_identity_at_init() -> None:
    torch.manual_seed(0)
    block = MonoResidual(32, 32, mode="absolute", activation="elu")
    x = torch.randn(8, 32)
    fx_rms = float(block.F(x).pow(2).mean().sqrt())
    assert fx_rms < 0.2  # F(x) ~= 0 at init => block ~= g_alpha * skip


def test_custom_F_is_not_near_zeroed() -> None:
    torch.manual_seed(0)
    custom = MonoLinear(32, 32, mode="absolute", activation="elu")
    before = float(custom.weight.detach().abs().sum())
    block = MonoResidual(32, 32, F=custom)
    after = float(block.F.weight.detach().abs().sum())  # type: ignore[union-attr]
    assert after == before  # untouched
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/torch/test_mono_residual_gate.py -v`
Expected: FAIL (`test_default_F_last_layer_is_near_zero_but_nonzero` — weight norm is a normal-init magnitude, not `<1.0`).

- [ ] **Step 3: Add `near_zero_out` to torch `MonoLinear` and use it in `MonoResidual`**

In `mononet/torch/layers.py`, add the module constant near the top (after imports):

```python
_NEAR_ZERO_SCALE = 1e-3
```

Add `near_zero_out: bool = False` to `MonoLinear.__init__`'s keyword-only params, and after the weight/bias are initialized (after the existing `self.bias = …` line ~99):

```python
        if near_zero_out:
            with torch.no_grad():
                self.weight.mul_(_NEAR_ZERO_SCALE)
                if self.bias is not None:
                    self.bias.zero_()
```

In `MonoResidual.__init__`'s default-`F` builder, pass `near_zero_out=True` to the last sub-layer. For `k == 1`:

```python
            if k == 1:
                self.F: nn.Module = MonoLinear(
                    in_features, units, mode=mode, activation=activation,
                    init=init, near_zero_out=True,
                )
```

For `k > 1`, build the intermediate layers normally and the final one with `near_zero_out=True`:

```python
            else:
                sub = [
                    MonoLinear(
                        in_features, units, mode=mode, activation=activation, init=init
                    )
                ]
                sub += [
                    MonoLinear(
                        units, units, mode=mode, activation=activation, init=init
                    )
                    for _ in range(k - 2)
                ]
                sub.append(
                    MonoLinear(
                        units, units, mode=mode, activation=activation,
                        init=init, near_zero_out=True,
                    )
                )
                self.F = nn.Sequential(*sub)
```

(Note: `k - 2` intermediate `units→units` layers + 1 near-zero last = `k - 1` after the first, preserving total `k`. For `k == 2` this yields `[input_layer, near_zero_last]`.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/torch/test_mono_residual_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Verify monotonicity still holds (torch)**

Run: `uv run pytest tests/torch/test_property_monotonic.py -v`
Expected: PASS (near-zero weights are a valid `|W|` point; block stays non-decreasing).

- [ ] **Step 6: Mirror in JAX**

Create `tests/jax/test_mono_residual_gate.py` (same three tests, `pytest.importorskip("jax")`, using `nnx`; read a weight via `last.weight[...]` and bias via `last.bias[...]`; the JAX `MonoResidual` stores the default stack in `nnx.Sequential`, so index the last built sub-layer — capture it as in Step 3's `sub[-1]`). Then in `mononet/jax/layers.py`:
- add `_NEAR_ZERO_SCALE = 1e-3`;
- add `near_zero_out: bool = False` to `MonoLinear.__init__`, and after the weight/bias `nnx.Param`s are created:

```python
        if near_zero_out:
            self.weight[...] = self.weight[...] * _NEAR_ZERO_SCALE
            if self.bias is not None:
                self.bias[...] = jnp.zeros_like(self.bias[...])
```

- in the `MonoResidual` default-`F` builder, build the last sub-layer with `near_zero_out=True` (same `k==1` / `k>1` structure as torch).

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax/test_mono_residual_gate.py tests/jax/test_property_monotonic.py -v` (skips if JAX not installed; CI covers it).

- [ ] **Step 7: Mirror in Keras**

Keras `MonoDense` creates weights in `build()`, not `__init__`. Create `tests/keras/test_mono_residual_gate.py` (same three tests, `pytest.importorskip("keras")`; build the block by calling it once on a dummy input so weights exist: `block(np.zeros((1, 32), dtype="float32"))`, then read `last.w`/`last.b`). Then in `mononet/keras/layers.py`:
- add `_NEAR_ZERO_SCALE = 1e-3`;
- add `near_zero_out: bool = False` to `MonoDense.__init__` (store `self.near_zero_out = near_zero_out`), and at the end of `MonoDense.build`, after `self.w`/`self.b` are created and before `super().build(...)`:

```python
        if self.near_zero_out:
            self.w.assign(self.w * _NEAR_ZERO_SCALE)
            if self.b is not None:
                self.b.assign(ops.zeros_like(self.b))
```

- in the `MonoResidual` default-`F` builder, build the last `MonoDense` with `near_zero_out=True` (same `k==1` / `k>1` split; for the `keras.Sequential` case build `k-1` normal + 1 near-zero).

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_mono_residual_gate.py tests/keras/test_property_monotonic.py -v` (skips if Keras not installed; CI covers it).

- [ ] **Step 8: Commit**

```bash
git add mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py \
        tests/torch/test_mono_residual_gate.py tests/jax/test_mono_residual_gate.py \
        tests/keras/test_mono_residual_gate.py
git commit -m "feat: near-zero init of default F last layer (fixes |W| frozen-weight trap)"
```

---

### Task 5: End-to-end regression — depth is used, F trains, no divergence (torch)

**Files:**
- Test: `tests/torch/test_deep_residual.py` (extend)

**Interfaces:**
- Consumes: the A+B default from Tasks 3-4; `benchmarks._common.init_diagnostics.build_residual_stack`, `synthetic_monotone`; `mononet.torch.MonoResidual`, `mononet.torch._kernels.gate`.
- Produces: the headline guard that deep default stacks open the gate, train `F`'s weights, and stay bounded — was verified red on `main`.

- [ ] **Step 1: Write the failing test**

Append to `tests/torch/test_deep_residual.py`:

```python
from mononet.torch import MonoResidual  # noqa: E402
from mononet.torch import _kernels  # noqa: E402


def _train_default_deep(depth: int = 16, epochs: int = 200, seed: int = 0):
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack("absolute", depth, 2)  # sub_depth=2, A+B defaults
    blocks = [m for m in net.modules() if isinstance(m, MonoResidual)]
    w0 = [float(_last_weight_abs_sum(b)) for b in blocks]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    g_beta = [float(_kernels.gate(b.beta_gate, b.beta)) for b in blocks]
    moved = sum(
        1 for b, w in zip(blocks, w0) if abs(float(_last_weight_abs_sum(b)) - w) > 1e-9
    )
    return loss_val, max(g_beta), moved, len(blocks)


def _last_weight_abs_sum(block: MonoResidual) -> float:
    from mononet.torch import MonoLinear

    f = block.F
    last = f if isinstance(f, MonoLinear) else f[-1]
    return float(last.weight.detach().abs().sum())


def test_deep_default_uses_depth() -> None:
    loss, max_g_beta, moved, n = _train_default_deep()
    # Trap-1 guard: gate opens (impossible under scaled_elu dead zone, g_beta~1e-3)
    assert max_g_beta > 0.1, f"gate did not open: max g_beta {max_g_beta}"
    # Trap-2 guard: F's last-layer weights actually train (exact-zero would freeze)
    assert moved == n, f"only {moved}/{n} blocks' F weights moved"
    # trains to a low floor
    assert loss < 0.3, f"deep default mse {loss}"
```

- [ ] **Step 2: Run to verify it passes (fix already in from Tasks 3-4)**

Run: `uv run pytest tests/torch/test_deep_residual.py -v`
Expected: PASS. (To confirm it is a real guard: it was verified red on `main` — `max g_beta ≈ 1e-3` fails the first assertion.)

- [ ] **Step 3: Commit**

```bash
git add tests/torch/test_deep_residual.py
git commit -m "test(torch): deep default MonoResidual opens gate and trains F"
```

---

### Task 6: Committed, reproducible evidence (trap instrumentation + ablation JSON)

**Files:**
- Create: `benchmarks/monoresidual_gate_trap.py`
- Modify: `benchmarks/monoresidual_gate_ablation.py` (emit JSON)
- Create: `benchmarks/results/monoresidual-gate/trap.json`, `benchmarks/results/monoresidual-gate/ablation.json`
- Test: `tests/benchmarks/test_monoresidual_gate_evidence.py` (NEW, smoke)

**Interfaces:**
- Consumes: `mononet.torch.MonoResidual`, `_kernels.gate`.
- Produces: two `python -m benchmarks.*` runnable modules that write committed JSON the docs render from.

- [ ] **Step 1: Write the trap-instrumentation script**

Create `benchmarks/monoresidual_gate_trap.py`: a self-contained module that builds a deep default-**pre-fix**-style stack (explicit `beta_gate="scaled_elu"` and a custom non-near-zero `F`) on the inline monotone teacher, trains a few hundred steps, and records per-step `g_beta` (min/max over blocks), `beta`, block-RMS, train/test MSE into a dict. Provide `main(out: Path)` that writes `benchmarks/results/monoresidual-gate/trap.json` and prints a short table. Model structure on `benchmarks/deep_residual_run.py` (argparse `--out`, `json.dump`, `# noqa: T201` prints). Reuse the teacher + `_Block` idiom from `benchmarks/monoresidual_gate_ablation.py` (import the shared helpers if convenient, else duplicate the small teacher fn).

- [ ] **Step 2: Make the ablation emit JSON**

In `benchmarks/monoresidual_gate_ablation.py`, change `_run` to *return* a result dict (`{a_mode, gate, train, test, g_beta_min, g_beta_max, f_moved, n_blocks}`) instead of only printing, and have `main` collect the six rows, print the table (as now), and — when given `--out` — `json.dump` them to `benchmarks/results/monoresidual-gate/ablation.json`. Keep the default (no `--out`) behaviour printing only, so the committed script stays runnable as before.

- [ ] **Step 3: Generate the committed JSON**

Run (on GPU if available, else CPU):
```bash
CUDA_VISIBLE_DEVICES=1 uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_ablation --out benchmarks/results/monoresidual-gate/ablation.json
CUDA_VISIBLE_DEVICES=1 uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_trap --out benchmarks/results/monoresidual-gate/trap.json
```
Expected: two JSON files written; ablation JSON's `nearzero/softplus` row has `g_beta_max > 0.1` and `f_moved == n_blocks`; trap JSON's final `g_beta_max ≈ 0`.

- [ ] **Step 4: Write a smoke test**

Create `tests/benchmarks/test_monoresidual_gate_evidence.py`:

```python
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "results" / "monoresidual-gate"


def test_ablation_json_shows_fix_beats_traps() -> None:
    rows = json.loads((RESULTS / "ablation.json").read_text())
    by = {(r["a_mode"], r["gate"]): r for r in rows}
    fix = by[("nearzero", "softplus")]
    assert fix["g_beta_max"] > 0.1
    assert fix["f_moved"] == fix["n_blocks"]
    # exact-zero freezes F's weights
    assert by[("exactzero", "softplus")]["f_moved"] == 0


def test_trap_json_shows_closed_gate() -> None:
    trap = json.loads((RESULTS / "trap.json").read_text())
    assert trap["final"]["g_beta_max"] < 0.05
```

(Match the exact JSON keys your Steps 1-2 emit; adjust the assertions to those keys.)

- [ ] **Step 5: Run the smoke test and lint**

Run: `uv run pytest tests/benchmarks/test_monoresidual_gate_evidence.py -v && uv run ruff check benchmarks/monoresidual_gate_trap.py benchmarks/monoresidual_gate_ablation.py && uv run mypy benchmarks/monoresidual_gate_trap.py`
Expected: PASS / clean.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/monoresidual_gate_trap.py benchmarks/monoresidual_gate_ablation.py \
        benchmarks/results/monoresidual-gate tests/benchmarks/test_monoresidual_gate_evidence.py
git commit -m "bench: committed trap + ablation evidence JSON for the gate fix"
```

---

### Task 7: Size-driven batch bands (Stage-2 infra)

**Files:**
- Modify: `benchmarks/_common/search_spaces.py` (`_LARGE_BATCH_DATASETS` → size rule; `suggest_config`)
- Modify: `benchmarks/_common/search.py` (`suggest_config(...)` call, ~line 113)
- Test: `tests/benchmarks/test_search_spaces.py`

**Interfaces:**
- Consumes: `bundle.X_train` (train-row count) at the `suggest_config` call site.
- Produces: `suggest_config(..., n_train: int)` selecting the large-batch band iff `n_train >= _LARGE_BATCH_THRESHOLD` (20_000). Removes the hardcoded name set.

- [ ] **Step 1: Update the test (failing)**

In `tests/benchmarks/test_search_spaces.py`, replace name-based expectations with size-based ones (read the file first to match its harness). Add:

```python
def test_batch_band_is_size_driven() -> None:
    import optuna
    from benchmarks._common.search_spaces import _BATCH_SIZES_LARGE, _BATCH_SIZES_SMALL, suggest_config

    def band(n_train: int) -> list[int]:
        seen: set[int] = set()
        study = optuna.create_study()
        for _ in range(60):
            t = study.ask()
            cfg = suggest_config(
                t, dataset="x", backend="torch", mode="absolute",
                residual=True, epochs=1, metric="mse", n_train=n_train,
            )
            seen.add(cfg.batch_size)
        return sorted(seen)

    assert set(band(50_000)).issubset(set(_BATCH_SIZES_LARGE))
    assert set(band(500)).issubset(set(_BATCH_SIZES_SMALL))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_search_spaces.py::test_batch_band_is_size_driven -v`
Expected: FAIL (`suggest_config` has no `n_train` parameter → `TypeError`).

- [ ] **Step 3: Implement the size rule**

In `benchmarks/_common/search_spaces.py`, replace:

```python
_LARGE_BATCH_DATASETS = frozenset({"loan", "blog"})
```

with:

```python
# Train-set-size threshold (rows) above which small batches make 50-epoch
# training intractable; the models are tiny so tuning is launch-bound, not
# capacity-bound. Derived from the loaded n_train so new datasets band
# automatically (no hand-maintained name set).
_LARGE_BATCH_THRESHOLD = 20_000
```

Add `n_train: int` as a keyword-only parameter of `suggest_config` (document it), and change the band selection:

```python
    batch_choices = (
        _BATCH_SIZES_LARGE if n_train >= _LARGE_BATCH_THRESHOLD else _BATCH_SIZES_SMALL
    )
```

Update the `:param dataset:` docstring to drop the `_LARGE_BATCH_DATASETS` reference and add `:param n_train:`.

- [ ] **Step 4: Update the caller**

In `benchmarks/_common/search.py`, the `suggest_config(` call (~line 113), add:

```python
            n_train=int(bundle.X_train.shape[0]),
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/benchmarks/test_search_spaces.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/_common/search_spaces.py benchmarks/_common/search.py tests/benchmarks/test_search_spaces.py
git commit -m "feat(bench): size-driven batch band (Stage-2 re-run infra)"
```

---

### Task 8: Rewrite the concepts docs — evidence-backed gate/skip design

**Files:**
- Modify: `docs/concepts/monotonic-residual.md`

**Interfaces:**
- Consumes: the committed `benchmarks/results/monoresidual-gate/*.json` from Task 6; the fixed layer behaviour from Tasks 3-4.
- Produces: the paper-grade page satisfying spec §7 (requirements, design choices, experiments, why A+B).

- [ ] **Step 1: Correct the refuted rationale**

In `docs/concepts/monotonic-residual.md`, the "Why the gates are shaped this way" subsection (currently ~lines 52-60) claims the `scaled_elu` `ε·exp(β/ε)` tail lets `F` "come online." Replace the `g_β` paragraph with the corrected two-traps account (spec §1/§3): (1) the `scaled_elu` dead-zone pins `g_β≈0` because a random `F` pushes `β` negative; the fix is `softplus` (dead-zone-free). Keep the `g_α` skip-gate paragraph as-is (it is correct).

- [ ] **Step 2: Add the "Requirements" and near-zero-init design**

Add a "Requirements for skip connections and gates" subsection (spec §7.1: skip monotone + near-identity-at-init + `g_α=1`; `F` monotone + `≈0` at init + weights-must-stay-trainable + `g_β>0` and able to open; positivity non-negotiable). Then add the near-zero-init design point (spec §7.2 Trap 2): "contribute ≈0 at init" is met by **near-zero init (scale `1e-3`), not exact-zero** — exact-zero is a `sign(0)=0` gradient fixed point under `|W|` that freezes `F` into a constant; note the stable scale band.

- [ ] **Step 3: Reframe the experiments and add trap + ablation**

Reframe the existing skip-K sweep paragraph as demonstrating **forward stability (non-divergence)**, explicitly *not* depth-utilisation. Add two subsections rendering the committed JSON: the **trap instrumentation** (`g_β≈0`, `F` idle) and the **A-vs-B ablation** (the six-row table from `ablation.json`, highlighting exact-zero `F-moved 0/16`, `off+softplus` divergence, and `nearzero+softplus` as the winner). Give each a one-line reproduce command:

```
uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_ablation
uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_trap
```

- [ ] **Step 4: "Why A+B" + before/after placeholder**

Add a short "Why A+B" synthesis (two traps ⇒ two fixes; neither alone suffices; monotonicity preserved). Add a "Depth on real data (before/after)" subsection with a `{note}` that it is populated by the Stage-2 re-run (#90/#99) — mirror the existing placeholder convention in `docs/benchmarks/large-dataset-screen.md`.

- [ ] **Step 5: Build the docs**

Run: `uv run pre-commit run docs --all-files` (or `./tools/build-docs.sh`)
Expected: `build succeeded`, no `-W` warnings.

- [ ] **Step 6: Commit**

```bash
git add docs/concepts/monotonic-residual.md
git commit -m "docs(concepts): evidence-backed gate/skip design (two traps, A+B)"
```

---

## Post-plan (out of band): Stage-2 benchmark re-run

Not a code task — executed by the controller after this plan lands (spec §8, Stage 2): re-run the standard flavor-comparison / deep-residual-accuracy table and the #90 screen across **all 10 datasets** (paper 5 + `adult`, `taiwan`, `polish`, `german`, `lc`), plus the #99 probe, on **both GPUs** (`n_jobs=1` per process), using the size-driven batch bands from Task 7. Then fill the docs before/after subsection (Task 8, Step 4) and update PRs #90 and #99.

## Self-Review notes

- **Spec coverage:** softplus token (T1-T2) ✓; default beta_gate (T3) ✓; near-zero init (T4) ✓; monotonicity preserved (T4 S5/S6/S7) ✓; depth-used + F-trains + no-divergence guards (T5) ✓; equivalence regen (T2) ✓; committed reproducible evidence (T6) ✓; size-driven batch band (T7) ✓; docs rewrite w/ requirements+design+experiments+why-A+B (T8) ✓; Stage-2 re-run captured as out-of-band controller work ✓.
- **Type consistency:** `near_zero_out` (bool kwarg) and `_NEAR_ZERO_SCALE` (1e-3) used identically across backends; `_last_linear`/`_last_weight_abs_sum` helpers local to their test files; `n_train` keyword added to `suggest_config` and passed from `bundle.X_train.shape[0]`.
- **`k>1` layer count:** builder is `[input] + (k-2)×[units→units] + [near-zero last]` = `k` layers total, matching the pre-fix count; `k==2` → `[input, near-zero-last]`.
