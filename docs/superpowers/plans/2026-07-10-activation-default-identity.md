# Default activation → identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Flip the default `activation` of the leaf monotone dense layers (and `MonoConfig`) from `"relu"` to `"identity"`, and make `activation` a mandatory, explicit choice for residual blocks (`MonoResidual` / `MonoResidualConfig`).

**Architecture:** Pure default/validation changes in the three backend layer modules (`mononet/{torch,jax,keras}/layers.py`) and the two config dataclasses (`mononet/core/config.py`). No kernel/reference math changes. Behavior is validated by new per-backend tests plus the unchanged cross-backend equivalence and monotonicity suites.

**Tech Stack:** Python 3.11+, PyTorch, Flax NNX (JAX), Keras 3, stdlib dataclasses, pytest, uv.

## Global Constraints

- Line length 88 (ruff); strict mypy; MyST field-list docstrings (`:param:`/`:returns:`/`:raises:`), types from signatures only.
- Do **not** touch kernels (`_kernels.py`), `mononet/core/reference.py`, the equivalence harness (`tests/equivalence/`), `convex_fraction`, the residual gates (`alpha_gate`/`beta_gate`), or `MonoInput`.
- `mononet/core/reference.py::monotonic_dense` already takes `activation` as a **required** parameter (no default) — spec §4's "reference default flip" is a no-op; make no change there.
- Branch: `feat/activation-default-identity` (already checked out).
- Full 3-backend verification requires the CPU `default` devcontainer (all backends installed). In a single-backend GPU container the other backends' tests `importorskip`-skip; that is expected, not a pass for them.
- Commit after each task. Commits may be unsigned in the container (no signing key); do not use `--no-verify` unless the pre-commit `docs` hook fails purely on a missing locale — if so, prefix the commit with `LC_ALL=C.UTF-8 LANG=C.UTF-8`.

---

### Task 1: Config defaults — `MonoConfig` → identity, `MonoResidualConfig.activation` required

**Files:**
- Modify: `mononet/core/config.py:25` (MonoConfig.activation), `mononet/core/config.py:84` (MonoResidualConfig.activation)
- Test: `tests/core/test_config.py`

**Interfaces:**
- Produces: `MonoConfig(units).activation.name == "identity"`; `MonoResidualConfig(units=...)` raises `TypeError` unless `activation=` is passed (keyword-only, required).

- [ ] **Step 1: Write the failing tests**

Append to `tests/core/test_config.py`:

```python
def test_monoconfig_default_activation_is_identity() -> None:
    from mononet.core.config import MonoConfig

    assert MonoConfig(units=8).activation.name == "identity"


def test_monoresidualconfig_requires_activation() -> None:
    import pytest

    from mononet.core.config import MonoResidualConfig

    with pytest.raises(TypeError):
        MonoResidualConfig(units=8)  # type: ignore[call-arg]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/core/test_config.py::test_monoconfig_default_activation_is_identity tests/core/test_config.py::test_monoresidualconfig_requires_activation -v`
Expected: FAIL (default is still `relu`; `MonoResidualConfig(units=8)` currently succeeds).

- [ ] **Step 3: Change the two defaults**

In `mononet/core/config.py`, MonoConfig (line ~25):

```python
    activation: ActivationSpec = field(
        default_factory=lambda: ActivationSpec("identity")
    )
```

In `mononet/core/config.py`, MonoResidualConfig (line ~84) — make it required, keyword-only:

```python
    activation: ActivationSpec = field(kw_only=True)
```

- [ ] **Step 4: Run the config suite**

Run: `uv run pytest tests/core/test_config.py -v`
Expected: PASS (new tests pass; the existing round-trip test at line ~40 already passes `activation=ActivationSpec("relu")`, so it is unaffected).

- [ ] **Step 5: Type-check**

Run: `uv run mypy mononet/core/config.py`
Expected: PASS (no errors).

- [ ] **Step 6: Commit**

```bash
git add mononet/core/config.py tests/core/test_config.py
git commit -m "feat(core): MonoConfig default activation identity; MonoResidualConfig activation required"
```

---

### Task 2: Leaf layers default → identity (`MonoLinear` / `MonoDense`, all backends)

**Files:**
- Modify: `mononet/torch/layers.py:66`, `mononet/jax/layers.py:75`, `mononet/keras/layers.py:58` (the leaf-layer `activation` default)
- Create: `tests/torch/test_default_activation.py`, `tests/jax/test_default_activation.py`, `tests/keras/test_default_activation.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `MonoLinear(...)` / `MonoDense(...)` with no `activation=` are affine (linear) maps.

- [ ] **Step 1: Write the failing tests (one file per backend)**

`tests/torch/test_default_activation.py`:

```python
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear  # noqa: E402


def test_default_activation_is_affine() -> None:
    # identity default => affine map => midpoint-preserving.
    layer = MonoLinear(4, 8, mode="switch")
    x1 = torch.randn(5, 4)
    x2 = torch.randn(5, 4)
    mid = layer((x1 + x2) / 2)
    avg = (layer(x1) + layer(x2)) / 2
    torch.testing.assert_close(mid, avg, rtol=1e-5, atol=1e-5)
```

`tests/jax/test_default_activation.py`:

```python
import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from mononet.jax import MonoLinear


def test_default_activation_is_affine() -> None:
    layer = MonoLinear(4, 8, mode="switch", rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    x1 = jnp.asarray(rng.standard_normal((5, 4)), dtype=jnp.float32)
    x2 = jnp.asarray(rng.standard_normal((5, 4)), dtype=jnp.float32)
    mid = layer((x1 + x2) / 2)
    avg = (layer(x1) + layer(x2)) / 2
    np.testing.assert_allclose(np.asarray(mid), np.asarray(avg), rtol=1e-5, atol=1e-5)
```

`tests/keras/test_default_activation.py`:

```python
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

pytest.importorskip("keras")

from mononet.keras import MonoDense


def test_default_activation_is_affine() -> None:
    layer = MonoDense(8, mode="switch")
    rng = np.random.default_rng(0)
    x1 = rng.standard_normal((5, 4)).astype("float32")
    x2 = rng.standard_normal((5, 4)).astype("float32")
    mid = np.asarray(layer((x1 + x2) / 2))
    avg = (np.asarray(layer(x1)) + np.asarray(layer(x2))) / 2
    np.testing.assert_allclose(mid, avg, rtol=1e-5, atol=1e-5)
```

- [ ] **Step 2: Run to verify they fail**

Run (per installed backend): `uv run pytest tests/torch/test_default_activation.py tests/jax/test_default_activation.py tests/keras/test_default_activation.py -v`
Expected: FAIL — the `relu` default makes the layers nonlinear, so midpoint ≠ average.

- [ ] **Step 3: Flip the leaf defaults**

In each file change the leaf-layer `__init__` signature line from `activation: ActivationSpec | str = "relu",` to `activation: ActivationSpec | str = "identity",`:
- `mononet/torch/layers.py:66` (MonoLinear)
- `mononet/jax/layers.py:75` (MonoLinear)
- `mononet/keras/layers.py:58` (MonoDense)

Leave the `MonoResidual` `activation` lines (torch:125, jax:146, keras:160) unchanged — Task 3 handles those.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/torch/test_default_activation.py tests/jax/test_default_activation.py tests/keras/test_default_activation.py -v`
Expected: PASS for every installed backend.

- [ ] **Step 5: Regression — property + public-api + equivalence still green**

Run: `uv run pytest tests/torch tests/jax tests/keras tests/equivalence -q`
Expected: PASS/skip. (Linear maps are still monotone; committed equivalence vectors set `activation` explicitly.) NOTE: `MonoResidual`-without-`activation` cases are still `relu`-default here and will only break after Task 3 — they should still pass at this point.

- [ ] **Step 6: Update docstrings**

In the three leaf layers, update the `:param activation:` line to note the new default, e.g. `:param activation: Base activation name or ActivationSpec (default "identity", i.e. a linear monotone map).`

- [ ] **Step 7: Commit**

```bash
git add mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py \
        tests/torch/test_default_activation.py tests/jax/test_default_activation.py \
        tests/keras/test_default_activation.py
git commit -m "feat(layers): default leaf activation to identity across backends"
```

---

### Task 3: `MonoResidual` — mandatory activation (Option A), all backends

**Files:**
- Modify: `mononet/torch/layers.py` (MonoResidual `__init__`, sig line ~125 + validation block after line ~141)
- Modify: `mononet/jax/layers.py` (MonoResidual `__init__`, sig line ~146 + validation after line ~157)
- Modify: `mononet/keras/layers.py` (MonoResidual `__init__`, sig line ~160 + reorder around line ~176/183)
- Modify (fix now-raising constructions): `tests/torch/test_public_api.py:31`, `tests/jax/test_public_api.py:31`, `tests/keras/test_public_api.py:37`, `tests/torch/test_property_monotonic.py:36`, and any other default-`F` `MonoResidual(...)` without `activation=` (see Step 5 grep).
- Test (new cases): `tests/torch/test_mono_residual_subdepth.py`, `tests/jax/test_mono_residual_subdepth.py`, `tests/keras/test_mono_residual_subdepth.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2.
- Produces: `MonoResidual` signature `activation: ActivationSpec | str | None = None` with the rule:
  - `F is None and activation is None` → `ValueError("activation is required when F is not provided")`
  - `F is not None and activation is not None` → `ValueError("pass either F or activation, not both")`

- [ ] **Step 1: Write the failing tests (append to each backend's `test_mono_residual_subdepth.py`)**

torch (`tests/torch/test_mono_residual_subdepth.py`):

```python
def test_default_F_without_activation_raises() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="activation is required"):
        MonoResidual(8, 8, mode="absolute")


def test_F_and_activation_together_raises() -> None:  # noqa: N802
    f = MonoLinear(8, 8, mode="absolute")
    with pytest.raises(ValueError, match="either F or activation"):
        MonoResidual(8, 8, F=f, activation="elu")
```

jax (`tests/jax/test_mono_residual_subdepth.py`):

```python
def test_default_F_without_activation_raises() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="activation is required"):
        MonoResidual(8, 8, mode="absolute", rngs=nnx.Rngs(0))


def test_F_and_activation_together_raises() -> None:  # noqa: N802
    f = MonoLinear(8, 8, mode="absolute", rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match="either F or activation"):
        MonoResidual(8, 8, F=f, activation="elu", rngs=nnx.Rngs(1))
```

keras (`tests/keras/test_mono_residual_subdepth.py`):

```python
def test_default_F_without_activation_raises() -> None:  # noqa: N802
    with pytest.raises(ValueError, match="activation is required"):
        MonoResidual(8, mode="absolute")


def test_F_and_activation_together_raises() -> None:  # noqa: N802
    f = MonoDense(8, mode="absolute")
    with pytest.raises(ValueError, match="either F or activation"):
        MonoResidual(8, F=f, activation="elu")
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/torch/test_mono_residual_subdepth.py -k "without_activation or together" -v` (and jax/keras equivalents)
Expected: FAIL — no such validation yet (default builds fine; `F`+`activation` is silently accepted).

- [ ] **Step 3a: torch — signature + validation**

`mononet/torch/layers.py`: change MonoResidual sig line `activation: ActivationSpec | str = "relu",` (line ~125) to `activation: ActivationSpec | str | None = None,`. Immediately after the existing `if F is not None and sub_depth is not None: raise ValueError("pass either F or sub_depth, not both")` block, add:

```python
        if F is None and activation is None:
            raise ValueError("activation is required when F is not provided")
        if F is not None and activation is not None:
            raise ValueError("pass either F or activation, not both")
```

Then in the `if F is None:` branch, add as its first line (for mypy narrowing):

```python
            assert activation is not None  # guaranteed by the check above
```

- [ ] **Step 3b: jax — signature + validation**

`mononet/jax/layers.py`: same signature change (line ~146) and the same two `raise` checks after the existing `if F is not None and sub_depth is not None:` block (after line ~157), plus `assert activation is not None` as the first line inside the `if F is None:` branch.

- [ ] **Step 3c: keras — signature + reorder**

`mononet/keras/layers.py`: change MonoResidual sig line (line ~160) to `activation: ActivationSpec | str | None = None,`. Keras currently sets `self.activation_name = _act_name(activation)` (line ~176) *before* validation, which breaks when `activation is None`. Replace the block from `self.activation_name = _act_name(activation)` through the end of the `F`/`else` build so validation runs first and `activation_name` is only derived when building the default `F`:

```python
        self.init_name = _init_name(init)
        self.alpha_gate = alpha_gate
        self.beta_gate = beta_gate
        if sub_depth is not None and sub_depth < 1:
            raise ValueError(f"sub_depth must be >= 1, got {sub_depth}")
        if F is not None and sub_depth is not None:
            raise ValueError("pass either F or sub_depth, not both")
        if F is None and activation is None:
            raise ValueError("activation is required when F is not provided")
        if F is not None and activation is not None:
            raise ValueError("pass either F or activation, not both")
        if F is not None:
            self.activation_name: str | None = None
            self.F: keras.layers.Layer = F
        else:
            assert activation is not None  # guaranteed by the check above
            self.activation_name = _act_name(activation)
            k = 2 if sub_depth is None else sub_depth
            if k == 1:
                self.F = MonoDense(units, mode=mode, activation=activation, init=init)
            else:
                self.F = keras.Sequential(
                    [
                        MonoDense(units, mode=mode, activation=activation, init=init)
                        for _ in range(k)
                    ]
                )
```

(Keep the earlier `self.units`/`self.mode` assignments as they are; only the `activation_name` line and the validation/F block move as shown. `get_config` at line ~245 continues to emit `self.activation_name`, now possibly `None` for the custom-`F` case — acceptable, since a custom `F` is not serialized anyway per the class docstring.)

- [ ] **Step 4: Run the new validation tests**

Run: `uv run pytest tests/torch/test_mono_residual_subdepth.py tests/jax/test_mono_residual_subdepth.py tests/keras/test_mono_residual_subdepth.py -v`
Expected: the two new tests per backend PASS; existing subdepth tests still PASS (they already pass `activation="elu"`, or raise on `sub_depth` which is checked first; `test_F_alone_is_used` passes `F=` with no `activation`, which is allowed).

- [ ] **Step 5: Fix now-raising default-`F` constructions elsewhere**

Find every `MonoResidual(...)` that builds the default `F` (no explicit `F=`) and omits `activation=`:

```bash
grep -rnE "MonoResidual\(" tests/ mononet/ benchmarks/ | grep -v "activation" | grep -v "F="
```

Known sites to fix by adding `activation="relu"` (keep them nonlinear as before the change):
- `tests/torch/test_public_api.py:31` → `t.MonoResidual(4, 4, mode="switch", activation="relu")`
- `tests/jax/test_public_api.py:31` → `j.MonoResidual(4, 4, mode="switch", activation="relu", rngs=nnx.Rngs(0))`
- `tests/keras/test_public_api.py:37` → `kmod.MonoResidual(4, mode="switch", activation="relu")`
- `tests/torch/test_property_monotonic.py:36` → `MonoResidual(3, 3, mode="switch", activation="relu")`
- Any additional hits from the grep (e.g. jax/keras property tests) → add `activation="relu"`.

(Do **not** add `activation` to lines that pass an explicit `F=` — that would now raise. `benchmarks/_common/model_builder.py` already passes `activation=cfg.activation`, so it needs no change.)

- [ ] **Step 6: Full regression per backend**

Run: `uv run pytest tests/torch tests/jax tests/keras tests/equivalence tests/core -q`
Expected: PASS/skip, no unexpected `ValueError`.

- [ ] **Step 7: Type-check**

Run: `uv run mypy mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py \
        tests/torch tests/jax tests/keras
git commit -m "feat(layers): require explicit activation for MonoResidual default F (Option A)"
```

---

### Task 4: Docs, examples, and CHANGELOG

**Files:**
- Modify: `README.md:42-43`, `docs/guides/pytorch.md:33-34`, `docs/guides/jax.md:33-34`, `docs/guides/keras.md:33-34`, `docs/concepts/monotonic-residual.md`
- Modify: `CHANGELOG.md` (under `## [Unreleased]`)

**Interfaces:** none (docs only).

- [ ] **Step 1: Fix the stacking examples so they stay nonlinear**

Under the new default a bare two-layer stack is linear. Add an explicit activation on the hidden layer; leave the read-out as the (now default) identity. Apply the analogous edit to each guide/README stack:

README.md / docs/guides/pytorch.md (the `Sequential([...])` example):

```python
    MonoLinear(4, 32, mode="switch", activation="relu"),
    MonoLinear(32, 1, mode="switch"),  # linear (identity) read-out
```

docs/guides/jax.md:

```python
    MonoLinear(4, 32, mode="switch", activation="relu", rngs=nnx.Rngs(0)),
    MonoLinear(32, 1, mode="switch", rngs=nnx.Rngs(1)),  # linear read-out
```

docs/guides/keras.md:

```python
    MonoDense(32, mode="switch", activation="relu"),
    MonoDense(1, mode="switch"),  # linear read-out
```

- [ ] **Step 2: Fix `MonoResidual` usages in concepts docs**

In `docs/concepts/monotonic-residual.md`, add `activation="relu"` (or the activation the surrounding text describes) to every `MonoResidual(...)` example that builds the default `F` (no explicit `F=`). Grep to find them:

```bash
grep -nE "MonoResidual\(" docs/concepts/monotonic-residual.md
```

- [ ] **Step 3: Add the CHANGELOG breaking-change entry**

Under `## [Unreleased]` in `CHANGELOG.md`, add a `### Changed` section (create it if absent):

```markdown
### Changed
- **BREAKING:** `MonoLinear` / `MonoDense` / `MonoConfig` now default
  `activation` to `"identity"` (was `"relu"`), matching `torch.nn.Linear`
  and `keras.layers.Dense`. Layers that relied on the implicit ReLU are now
  linear monotone maps — pass `activation="relu"` explicitly to restore the
  previous behavior.
- **BREAKING:** `MonoResidual` and `MonoResidualConfig` now require an explicit
  `activation` when the default `F` is built (a custom `F` must not also pass
  `activation`), preventing a silently-linear residual branch.
```

- [ ] **Step 4: Build the docs**

Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html`
Expected: `build succeeded`, no warnings.

- [ ] **Step 5: Full pre-commit gate**

Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run pre-commit run --all-files`
Expected: all hooks Passed/Skipped.

- [ ] **Step 6: Commit**

```bash
git add README.md docs/guides CHANGELOG.md docs/concepts/monotonic-residual.md
git commit -m "docs: update examples for identity default + CHANGELOG breaking notes"
```

---

## Self-Review

**Spec coverage:**
- §3a leaf default → identity → Task 2 (+ MonoConfig in Task 1). Reference no-op noted in Global Constraints. ✓
- §3b MonoResidual mandatory activation (Option A) → Task 3; MonoResidualConfig required → Task 1. ✓
- §5 migration (CHANGELOG breaking notes, `activation="relu"` recovery) → Task 4. Version bump left to maintainer (spec §8, out of scope). ✓
- §6 tests (affine leaf, residual validation, config TypeError, equivalence/property unchanged) → Tasks 1–3; docs audit + strict build → Task 4. ✓
- §7 out-of-scope items untouched (kernels/reference/gates/convex_fraction/MonoInput) → enforced in Global Constraints. ✓

**Placeholder scan:** No TBD/TODO; every code step shows concrete code; every command has expected output. ✓

**Type consistency:** New signature `activation: ActivationSpec | str | None = None` is used identically across Tasks 3a/3b/3c; the two `ValueError` messages ("activation is required when F is not provided", "pass either F or activation, not both") match between implementation steps and the `pytest.raises(match=...)` tests. `field(kw_only=True)` (Task 1) matches the `TypeError` test. ✓
