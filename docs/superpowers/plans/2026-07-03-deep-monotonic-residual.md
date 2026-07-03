# Deep Monotonic Networks via `MonoResidual.sub_depth` — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `MonoResidual` with a `sub_depth` parameter (a monotone residual skip every K layers) so deep (target 32) monotonic nets train, across torch/JAX/Keras — with a monotonicity property test, a committed skip-K trainability sweep, and a paper-grade docs page.

**Architecture:** `MonoResidual`'s default `F` becomes a `sub_depth`-deep stack of `MonoLinear`/`MonoDense` (via each framework's `Sequential`); `sub_depth=1` is byte-equivalent to today's single-layer default. Dual gates + near-identity warm start are unchanged, so a uniform-width `Sequential` of these blocks starts ≈ identity and trains at depth. No new composed model class; the deep net is a documented `Sequential` recipe.

**Tech Stack:** Python 3.11, PyTorch / JAX (Flax NNX) / Keras 3, NumPy, pytest, Sphinx + myst-nb.

## Global Constraints

- Branch **`feat/deep-mono-residual`** (already created off main); never commit to `main`.
- Commit **UNSIGNED** during subagent execution (`git -c commit.gpgsign=false commit`); controller re-signs the whole branch before push.
- Per-task gates: `uv run ruff check`, `uv run ruff format --check`, `uv run --group bench mypy`, the task's pytest (with matching `MONONET_TEST_BACKEND` for backend tasks). Final: `uv run pre-commit run --all-files --hook-stage manual` + `./tools/build-docs.sh`.
- No Pydantic; stdlib dataclasses; MyST field-list docstrings (`:param:`/`:returns:`/`:raises:`, no `:type:`); ruff line-88; strict mypy; preserve lazy backend imports; `benchmarks/` never in the wheel; result JSON with a trailing newline; never commit `*.db`/`*.jsonl`.
- **Design invariants:** `sub_depth: int | None = None` resolves to a **default of 2** (a skip every 2 layers). The sentinel `None` = "caller didn't set it". Let `k = 2 if sub_depth is None else sub_depth`. For `k == 1`, the default `F` is a single `MonoLinear`/`MonoDense` (byte-equivalent to the *legacy* default `F`, NOT wrapped in `Sequential`). For `k > 1`, the default `F` is `MonoLinear(in,units)` then `(k-1)× MonoLinear(units,units)` sharing `mode`/`activation`/`init` (`MonoDense` for Keras) via the framework's `Sequential`. **This changes the default `MonoResidual`** — its default `F` is now a 2-layer stack. Validation: `F is not None and sub_depth is not None` → `ValueError` (F alone is fine — F used, default ignored); `sub_depth is not None and sub_depth < 1` → `ValueError`. No change to `MonoLinear`/`MonoInput`/kernels or the `switch`/`absolute` math; the stateless-kernel equivalence harness is untouched. Near-identity warm start (`alpha=beta=0`) unchanged (warm start is depth-independent, so existing MonoResidual tests still pass).

---

### Task 1: Torch `MonoResidual.sub_depth`

**Files:**
- Modify: `mononet/torch/layers.py` (`MonoResidual.__init__`)
- Test: `tests/torch/test_mono_residual_subdepth.py`

**Interfaces:**
- Produces: `MonoResidual(in_features, units, *, F=None, mode=..., activation=..., alpha_gate=..., beta_gate=..., init=None, sub_depth=None)`. Default (`sub_depth is None` → k=2) → `self.F` is `nn.Sequential` of 2 `MonoLinear`; `sub_depth=1` → single `MonoLinear`; `sub_depth=K>1` → `nn.Sequential` of K; `F`+explicit `sub_depth` or `sub_depth<1` raise `ValueError`; `F` alone → uses `F`.

- [ ] **Step 1: Write the failing tests** — `tests/torch/test_mono_residual_subdepth.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from mononet.torch import MonoLinear, MonoResidual


def test_default_builds_two_monolinears() -> None:
    layer = MonoResidual(8, 8, mode="absolute", activation="elu")  # default sub_depth -> 2
    assert isinstance(layer.F, torch.nn.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F) == 2


def test_subdepth_builds_k_monolinears() -> None:
    layer = MonoResidual(8, 8, mode="absolute", activation="elu", sub_depth=3)
    assert isinstance(layer.F, torch.nn.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F) == 3


def test_subdepth1_is_single_monolinear() -> None:
    layer = MonoResidual(8, 8, mode="absolute", activation="elu", sub_depth=1)
    assert isinstance(layer.F, MonoLinear)


def test_F_alone_is_used() -> None:  # F without sub_depth must NOT raise
    f = MonoLinear(8, 8, mode="absolute")
    layer = MonoResidual(8, 8, F=f)
    assert layer.F is f


def test_F_and_explicit_subdepth_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, 8, F=MonoLinear(8, 8, mode="absolute"), sub_depth=2)


def test_subdepth_below_one_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, 8, mode="absolute", sub_depth=0)


def _nondecreasing(layer: torch.nn.Module, in_f: int) -> None:
    torch.manual_seed(1)
    with torch.no_grad():  # exercise non-trivial gates (must hold for any params)
        layer.alpha.fill_(0.3)
        layer.beta.fill_(0.7)
    x = torch.randn(64, in_f, dtype=torch.float64)
    y0 = layer(x)
    for i in range(in_f):
        xp = x.clone()
        xp[:, i] += 0.5
        assert (layer(xp) - y0).min().item() >= -1e-9


def test_monotone_identity_skip() -> None:
    torch.manual_seed(0)
    _nondecreasing(MonoResidual(6, 6, mode="absolute", activation="elu", sub_depth=2).double(), 6)


def test_monotone_projection_skip() -> None:
    torch.manual_seed(0)
    _nondecreasing(MonoResidual(6, 4, mode="switch", activation="elu", sub_depth=2).double(), 6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_mono_residual_subdepth.py -q`
Expected: FAIL — `MonoResidual` has no `sub_depth` keyword (`TypeError`).

- [ ] **Step 3: Implement** — in `mononet/torch/layers.py`, add `sub_depth: int | None = None,` to `MonoResidual.__init__` (after `init`), a `:param sub_depth:` docstring line (default 2; `1` = legacy single layer), and replace the `F`-construction block:

```python
        if F is None:
            self.F: nn.Module = MonoLinear(
                in_features, units, mode=mode, activation=activation, init=init
            )
        elif callable(F) and not isinstance(F, nn.Module):
            self.F = F(units)
        else:
            self.F = F
```

with:

```python
        if sub_depth is not None and sub_depth < 1:
            raise ValueError(f"sub_depth must be >= 1, got {sub_depth}")
        if F is not None and sub_depth is not None:
            raise ValueError("pass either F or sub_depth, not both")
        if F is None:
            k = 2 if sub_depth is None else sub_depth
            if k == 1:
                self.F: nn.Module = MonoLinear(
                    in_features, units, mode=mode, activation=activation, init=init
                )
            else:
                sub = [
                    MonoLinear(
                        in_features, units, mode=mode, activation=activation, init=init
                    )
                ]
                sub += [
                    MonoLinear(units, units, mode=mode, activation=activation, init=init)
                    for _ in range(k - 1)
                ]
                self.F = nn.Sequential(*sub)
        elif callable(F) and not isinstance(F, nn.Module):
            self.F = F(units)
        else:
            self.F = F
```

- [ ] **Step 4: Run to verify it passes**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_mono_residual_subdepth.py tests/torch/test_public_api.py -q`
Expected: PASS (new file green; existing torch API tests still green — `sub_depth=1` path unchanged).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/torch/layers.py tests/torch/test_mono_residual_subdepth.py && uv run ruff format --check mononet/torch/layers.py tests/torch/test_mono_residual_subdepth.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/torch/layers.py tests/torch/test_mono_residual_subdepth.py
git -c commit.gpgsign=false commit -m "feat(torch): MonoResidual sub_depth (skip every K layers) + monotonicity tests"
```

---

### Task 2: JAX `MonoResidual.sub_depth`

**Files:**
- Modify: `mononet/jax/layers.py` (`MonoResidual.__init__`)
- Test: `tests/jax/test_mono_residual_subdepth.py`

**Interfaces:**
- Produces: JAX `MonoResidual(in_features, units, *, F=None, ..., init=None, sub_depth=None, rngs)`; default (`None`→k=2) → `self.F` is `nnx.Sequential` of 2 `MonoLinear`; `sub_depth=1` → single; `sub_depth=K>1` → `nnx.Sequential` of K; same validation as torch (F+explicit raises; F alone OK; sub_depth<1 raises).

- [ ] **Step 1: Write the failing tests** — `tests/jax/test_mono_residual_subdepth.py`:

```python
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from mononet.jax import MonoLinear, MonoResidual


def test_default_builds_two_monolinears() -> None:
    layer = MonoResidual(8, 8, mode="absolute", activation="elu", rngs=nnx.Rngs(0))
    assert isinstance(layer.F, nnx.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F.layers) == 2


def test_subdepth_builds_k_monolinears() -> None:
    layer = MonoResidual(8, 8, mode="absolute", activation="elu", sub_depth=3, rngs=nnx.Rngs(0))
    assert isinstance(layer.F, nnx.Sequential)
    assert sum(isinstance(m, MonoLinear) for m in layer.F.layers) == 3


def test_subdepth1_is_single_monolinear() -> None:
    layer = MonoResidual(8, 8, mode="absolute", sub_depth=1, rngs=nnx.Rngs(0))
    assert isinstance(layer.F, MonoLinear)


def test_F_alone_is_used() -> None:
    f = MonoLinear(8, 8, mode="absolute", rngs=nnx.Rngs(0))
    layer = MonoResidual(8, 8, F=f, rngs=nnx.Rngs(0))
    assert layer.F is f


def test_F_and_explicit_subdepth_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(
            8, 8, F=MonoLinear(8, 8, mode="absolute", rngs=nnx.Rngs(0)),
            sub_depth=2, rngs=nnx.Rngs(0),
        )


def test_subdepth_below_one_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, 8, mode="absolute", sub_depth=0, rngs=nnx.Rngs(0))


def _nondecreasing(layer: nnx.Module, in_f: int) -> None:
    layer.alpha.value = jnp.array(0.3)
    layer.beta.value = jnp.array(0.7)
    import jax
    x = jax.random.normal(jax.random.key(1), (64, in_f))
    y0 = layer(x)
    for i in range(in_f):
        xp = x.at[:, i].add(0.5)
        assert float(jnp.min(layer(xp) - y0)) >= -1e-4


def test_monotone_identity_skip() -> None:
    _nondecreasing(MonoResidual(6, 6, mode="absolute", activation="elu", sub_depth=2, rngs=nnx.Rngs(0)), 6)


def test_monotone_projection_skip() -> None:
    _nondecreasing(MonoResidual(6, 4, mode="switch", activation="elu", sub_depth=2, rngs=nnx.Rngs(0)), 6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax/test_mono_residual_subdepth.py -q`
Expected: FAIL — no `sub_depth` keyword.

- [ ] **Step 3: Implement** — in `mononet/jax/layers.py`, add `sub_depth: int | None = None,` to `MonoResidual.__init__` (after `init`, before `rngs`), a `:param sub_depth:` docstring line (default 2; `1` = legacy single layer), and replace the `F`-construction block:

```python
        if F is None:
            self.F: nnx.Module = MonoLinear(
                in_features,
                units,
                mode=mode,
                activation=activation,
                init=init,
                rngs=rngs,
            )
        elif callable(F) and not isinstance(F, nnx.Module):
            self.F = F(units)
        else:
            self.F = F
```

with:

```python
        if sub_depth is not None and sub_depth < 1:
            raise ValueError(f"sub_depth must be >= 1, got {sub_depth}")
        if F is not None and sub_depth is not None:
            raise ValueError("pass either F or sub_depth, not both")
        if F is None:
            k = 2 if sub_depth is None else sub_depth
            if k == 1:
                self.F: nnx.Module = MonoLinear(
                    in_features, units, mode=mode, activation=activation,
                    init=init, rngs=rngs,
                )
            else:
                sub = [
                    MonoLinear(
                        in_features, units, mode=mode, activation=activation,
                        init=init, rngs=rngs,
                    )
                ]
                sub += [
                    MonoLinear(
                        units, units, mode=mode, activation=activation,
                        init=init, rngs=rngs,
                    )
                    for _ in range(k - 1)
                ]
                self.F = nnx.Sequential(*sub)
        elif callable(F) and not isinstance(F, nnx.Module):
            self.F = F(units)
        else:
            self.F = F
```

- [ ] **Step 4: Run to verify it passes**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/jax/test_mono_residual_subdepth.py tests/jax -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/jax/layers.py tests/jax/test_mono_residual_subdepth.py && uv run ruff format --check mononet/jax/layers.py tests/jax/test_mono_residual_subdepth.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/jax/layers.py tests/jax/test_mono_residual_subdepth.py
git -c commit.gpgsign=false commit -m "feat(jax): MonoResidual sub_depth (skip every K layers) + monotonicity tests"
```

---

### Task 3: Keras `MonoResidual.sub_depth`

**Files:**
- Modify: `mononet/keras/layers.py` (`MonoResidual.__init__`)
- Test: `tests/keras/test_mono_residual_subdepth.py`

**Interfaces:**
- Produces: Keras `MonoResidual(units, *, F=None, ..., init=None, sub_depth=None)`; default (`None`→k=2) → `self.F` is `keras.Sequential` of 2 `MonoDense`; `sub_depth=1` → single `MonoDense`; `sub_depth=K>1` → `keras.Sequential` of K; same validation (F+explicit raises; F alone OK; sub_depth<1 raises).

- [ ] **Step 1: Write the failing tests** — `tests/keras/test_mono_residual_subdepth.py`:

```python
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np
import pytest

pytest.importorskip("keras")

import keras
from keras import ops

from mononet.keras import MonoDense, MonoResidual


def test_default_builds_two_monodense() -> None:
    layer = MonoResidual(8, mode="absolute", activation="elu")  # default sub_depth -> 2
    assert isinstance(layer.F, keras.Sequential)
    assert sum(isinstance(m, MonoDense) for m in layer.F.layers) == 2


def test_subdepth_builds_k_monodense() -> None:
    layer = MonoResidual(8, mode="absolute", activation="elu", sub_depth=3)
    assert isinstance(layer.F, keras.Sequential)
    assert sum(isinstance(m, MonoDense) for m in layer.F.layers) == 3


def test_subdepth1_is_single_monodense() -> None:
    layer = MonoResidual(8, mode="absolute", sub_depth=1)
    assert isinstance(layer.F, MonoDense)


def test_F_alone_is_used() -> None:
    f = MonoDense(8, mode="absolute")
    layer = MonoResidual(8, F=f)
    assert layer.F is f


def test_F_and_explicit_subdepth_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, F=MonoDense(8, mode="absolute"), sub_depth=2)


def test_subdepth_below_one_raises() -> None:
    with pytest.raises(ValueError, match="sub_depth"):
        MonoResidual(8, mode="absolute", sub_depth=0)


def _nondecreasing(units: int, in_f: int, mode: str) -> None:
    layer = MonoResidual(units, mode=mode, activation="elu", sub_depth=2)
    x = ops.convert_to_tensor(np.random.default_rng(1).standard_normal((64, in_f)).astype("float32"))
    layer(x)  # build
    layer.beta.assign(ops.convert_to_tensor(0.7, dtype=layer.beta.dtype))
    y0 = ops.convert_to_numpy(layer(x))
    for i in range(in_f):
        xp = np.array(ops.convert_to_numpy(x)); xp[:, i] += 0.5
        y1 = ops.convert_to_numpy(layer(ops.convert_to_tensor(xp)))
        assert float((y1 - y0).min()) >= -1e-3


def test_monotone_identity_skip() -> None:
    _nondecreasing(6, 6, "absolute")


def test_monotone_projection_skip() -> None:
    _nondecreasing(4, 6, "switch")
```

- [ ] **Step 2: Run to verify it fails**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_mono_residual_subdepth.py -q`
Expected: FAIL — no `sub_depth` keyword.

- [ ] **Step 3: Implement** — in `mononet/keras/layers.py`, add `sub_depth: int | None = None,` to `MonoResidual.__init__` (after `init`, before `**kwargs`), a `:param sub_depth:` docstring line (default 2; `1` = legacy single layer), and replace:

```python
        self.F = (
            F
            if F is not None
            else MonoDense(units, mode=mode, activation=activation, init=init)
        )
```

with:

```python
        if sub_depth is not None and sub_depth < 1:
            raise ValueError(f"sub_depth must be >= 1, got {sub_depth}")
        if F is not None and sub_depth is not None:
            raise ValueError("pass either F or sub_depth, not both")
        if F is not None:
            self.F: keras.layers.Layer = F
        else:
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

- [ ] **Step 4: Run to verify it passes**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/keras/test_mono_residual_subdepth.py tests/keras -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check mononet/keras/layers.py tests/keras/test_mono_residual_subdepth.py && uv run ruff format --check mononet/keras/layers.py tests/keras/test_mono_residual_subdepth.py && uv run --group bench mypy`
Expected: clean. (If mypy flags the `self.F` union re-annotation, annotate `self.F: keras.layers.Layer` on the first assignment only, as shown.)

- [ ] **Step 6: Commit (unsigned)**

```bash
git add mononet/keras/layers.py tests/keras/test_mono_residual_subdepth.py
git -c commit.gpgsign=false commit -m "feat(keras): MonoResidual sub_depth (skip every K layers) + monotonicity tests"
```

---

### Task 4: Diagnostics helper + skip-K sweep runner

**Files:**
- Modify: `benchmarks/_common/init_diagnostics.py` (add `build_residual_stack`)
- Create: `benchmarks/deep_residual_run.py`
- Create: `benchmarks/results/deep-residual/.gitignore`
- Test: `tests/benchmarks/test_deep_residual_run.py`

**Interfaces:**
- Consumes: torch `MonoLinear`, `MonoResidual(sub_depth=…)` (Task 1).
- Produces: `build_residual_stack(mode: str, depth: int, sub_depth: int | None, *, width: int = 32) -> nn.Module` — `sub_depth=None` → plain baseline (`depth` `MonoLinear(W,W)`); else uniform-width body of `depth // sub_depth` `MonoResidual(W, W, sub_depth=sub_depth)` blocks, sandwiched by `MonoLinear(8→W)` / `MonoLinear(W→1)`. `benchmarks/deep_residual_run.py:main()` writes `benchmarks/results/deep-residual/trainability.json`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_deep_residual_run.py`:

```python
import pytest

pytest.importorskip("torch")

import torch.nn as nn

from benchmarks._common.init_diagnostics import build_residual_stack
from mononet.torch import MonoResidual


def test_build_residual_stack_block_count() -> None:
    net = build_residual_stack("absolute", depth=8, sub_depth=2, width=16)
    assert sum(isinstance(m, MonoResidual) for m in net) == 4  # 8 // 2


def test_build_residual_stack_plain_has_no_residual() -> None:
    net = build_residual_stack("absolute", depth=8, sub_depth=None, width=16)
    assert sum(isinstance(m, MonoResidual) for m in net) == 0
    assert isinstance(net, nn.Sequential)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_deep_residual_run.py -q`
Expected: FAIL — `build_residual_stack` does not exist.

- [ ] **Step 3: Implement `build_residual_stack`** — append to `benchmarks/_common/init_diagnostics.py`:

```python
from mononet.torch import MonoResidual  # add to the existing mononet.torch import line


def build_residual_stack(
    mode: str, depth: int, sub_depth: int | None, *, width: int = 32
) -> nn.Module:
    """Uniform-width monotone stack; residual (skip every ``sub_depth``) or plain.

    :param mode: ``switch`` or ``absolute``.
    :param depth: Number of hidden ``W->W`` monotone layers.
    :param sub_depth: Layers per residual block; ``None`` builds a plain (no-skip) stack.
    :param width: Uniform hidden width ``W``.
    :returns: An ``nn.Sequential`` mapping ``(batch, 8) -> (batch, 1)``.
    """
    layers: list[nn.Module] = [MonoLinear(8, width, mode=mode, activation="elu")]
    if sub_depth is None:
        layers += [
            MonoLinear(width, width, mode=mode, activation="elu") for _ in range(depth)
        ]
    else:
        layers += [
            MonoResidual(width, width, mode=mode, activation="elu", sub_depth=sub_depth)
            for _ in range(depth // sub_depth)
        ]
    layers.append(MonoLinear(width, 1, mode=mode, activation="elu"))
    return nn.Sequential(*[m.double() for m in layers])
```

- [ ] **Step 4: Create `benchmarks/deep_residual_run.py`** (the sweep runner):

```python
"""Skip-K trainability + conditioning sweep for deep monotone residual stacks.

Writes ``benchmarks/results/deep-residual/trainability.json`` (committed; read by
``docs/concepts/monotonic-residual.md``). Repo-only; never shipped in the wheel.

Run: ``uv run --extra torch --group bench python -m benchmarks.deep_residual_run``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmarks._common.init_diagnostics import build_residual_stack, synthetic_monotone

_MODES = ("absolute", "switch")
_DEPTHS = (4, 8, 16, 32)
_KS: tuple[int | None, ...] = (None, 1, 2, 4, 8)
_CAP = 1.0e6


def _run(mode: str, depth: int, sub_depth: int | None, *, epochs: int = 300, seed: int = 0) -> tuple[float, float]:
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack(mode, depth, sub_depth)
    xg = x.clone().requires_grad_(True)
    net(xg).sum().backward()  # type: ignore[no-untyped-call]
    assert xg.grad is not None
    gnorm = float(xg.grad.norm() / xg.shape[0] ** 0.5)
    net = build_residual_stack(mode, depth, sub_depth)  # fresh for training
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    mse = _CAP if (not np.isfinite(loss_val) or loss_val > _CAP) else loss_val
    return mse, min(gnorm, _CAP)


def main() -> None:
    """Run the sweep and write the committed results JSON."""
    rows: list[dict[str, float | str | int]] = []
    for mode in _MODES:
        for depth in _DEPTHS:
            for k in _KS:
                if k is not None and k > depth:
                    continue
                mse, gnorm = _run(mode, depth, k)
                rows.append(
                    {
                        "mode": mode,
                        "depth": depth,
                        "skip_k": "plain" if k is None else k,
                        "final_train_mse": round(mse, 4),
                        "init_grad_norm": float(f"{gnorm:.4g}"),
                    }
                )
                print(f"{mode:9} d{depth:<2} K={str(k):5} mse={mse:.4f} g={gnorm:.3e}")  # noqa: T201
    out = Path(__file__).resolve().parent / "results" / "deep-residual" / "trainability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {out}")  # noqa: T201


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Create `benchmarks/results/deep-residual/.gitignore`**

```
*.db
*.jsonl
```

- [ ] **Step 6: Run tests + lint + types**

Run: `uv run pytest tests/benchmarks/test_deep_residual_run.py -q && uv run ruff check benchmarks/deep_residual_run.py benchmarks/_common/init_diagnostics.py tests/benchmarks/test_deep_residual_run.py && uv run ruff format --check benchmarks/deep_residual_run.py benchmarks/_common/init_diagnostics.py && uv run --group bench mypy`
Expected: 2 passed; ruff/format/mypy clean. (Do NOT run the full sweep here — that's the controller phase.)

- [ ] **Step 7: Commit (unsigned)**

```bash
git add benchmarks/_common/init_diagnostics.py benchmarks/deep_residual_run.py benchmarks/results/deep-residual/.gitignore tests/benchmarks/test_deep_residual_run.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): residual-stack builder + skip-K sweep runner"
```

---

### Task 5: Fast deep-trainability regression (torch)

**Files:**
- Create: `tests/torch/test_deep_residual.py`

**Interfaces:**
- Consumes: `build_residual_stack` (Task 4), `synthetic_monotone` (existing).

- [ ] **Step 1: Write the test** — `tests/torch/test_deep_residual.py`:

```python
import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn

from benchmarks._common.init_diagnostics import build_residual_stack, synthetic_monotone


def _final_mse(sub_depth: int | None, *, depth: int = 32, epochs: int = 200, seed: int = 0) -> float:
    torch.manual_seed(seed)
    x_np, y_np = synthetic_monotone(512, 8, seed=seed)
    x = torch.tensor(x_np, dtype=torch.float64)
    y = torch.tensor(y_np, dtype=torch.float64).unsqueeze(1)
    net = build_residual_stack("absolute", depth, sub_depth)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_val = float("inf")
    for _ in range(epochs):
        opt.zero_grad()
        loss = nn.functional.mse_loss(net(x), y)
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        loss_val = float(loss.detach())
    return loss_val


def test_deep_residual_trains_where_plain_fails() -> None:
    # depth-32 absolute: sub_depth=2 residual trains (~0.10 measured); plain diverges.
    residual = _final_mse(2)
    plain = _final_mse(None)
    assert residual < 0.3, f"residual d32 mse {residual}"
    assert plain > 1.0, f"plain d32 mse {plain}"
```

- [ ] **Step 2: Run to verify it passes** (regression guard; the impl is already in)

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/torch/test_deep_residual.py -q`
Expected: PASS (residual ~0.10 < 0.3; plain diverged > 1.0). No red-first — it guards existing behaviour.

- [ ] **Step 3: Lint + types**

Run: `uv run ruff check tests/torch/test_deep_residual.py && uv run ruff format --check tests/torch/test_deep_residual.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 4: Commit (unsigned)**

```bash
git add tests/torch/test_deep_residual.py
git -c commit.gpgsign=false commit -m "test(torch): depth-32 sub_depth=2 residual trains where plain diverges"
```

---

### Task 6: Paper-grade docs page

**Files:**
- Create: `docs/concepts/monotonic-residual.md`
- Modify: `docs/concepts/index.md` (add to the toctree)

**Interfaces:**
- Consumes: committed `benchmarks/results/deep-residual/trainability.json` (Task 7 writes it; page uses a missing-results guard so the docs build is green before then).

- [ ] **Step 1: Create `docs/concepts/monotonic-residual.md`**

A MyST page with these sections. Transcribe the **theory faithfully from the spec §3** (this is the essential, paper-grade content — copy §3.1.1 gate rationale, §3.2 monotonicity proof for *both* size cases, §3.3 warm-start + K analysis):

````markdown
# Deep monotonic networks with residual skips

## Motivation
Deep *plain* monotone stacks fail to train: `|W|`'s all-positive weights make layer outputs
strongly correlated, so variance compounds with depth (both `absolute` and `switch` diverge by
depth ≥ 8). Static init cannot fix this (see the absolute-init results). Residual skips do.

## Construction
`MonoResidual` computes `y = g_α(α)·skip(x) + g_β(β)·F(x)`, with `sub_depth=K` making `F` a
K-deep monotone stack. A deep monotone net is a uniform-width `Sequential`:

```python
from mononet.torch import MonoLinear, MonoResidual
import torch.nn as nn

W = 32
net = nn.Sequential(
    MonoLinear(n_in, W, mode="absolute", activation="elu"),
    *[MonoResidual(W, W, sub_depth=2, mode="absolute", activation="elu") for _ in range(15)],
    MonoLinear(W, 1, mode="absolute", activation="elu"),
)  # ~depth 32; uniform width => identity skips
```

## Theory
### Why the gates are shaped this way
<transcribe spec §3.1.1 verbatim: positivity => monotonicity invariant; g_α = elu(α)+1 = 1 at
init (identity warm start; why not sigmoid/exp/softplus); g_β = scaled_elu ≈ ε at init (F off),
exp tail avoids the dead-ReLU-gate zero-gradient trap, linear/unbounded for β>0>

### Monotonicity (both size cases)
<transcribe spec §3.2 verbatim: F non-decreasing (absolute & switch); identity skip (in==out)
and positive exp-projection skip (in≠out); positive-weighted sum of non-decreasing maps is
non-decreasing; hard invariant under free optimization; MonoInput handles direction; composition>

### Why depth becomes trainable, and the role of K
<transcribe spec §3.3 verbatim: near-identity warm start => stack ≈ identity => signal/gradient
flow; F is a K-deep sub-stack that blows up by depth ~4–8, so K must be ≤ that; K=2 balances
conditioning vs expressiveness>

## Experiments
Skip-K trainability sweep (synthetic monotone target, 300-epoch Adam; final train MSE, `<0.5` =
learns) and init input-gradient norm (conditioning). Reproduce:
`uv run --extra torch --group bench python -m benchmarks.deep_residual_run`.

<code cell / rendered table from the committed JSON, with a missing-results guard>

## Real-dataset accuracy (forthcoming)
Stage 2 will report whether the now-trainable depth improves test metrics on real datasets vs
the shallow tuned flavors. *(Results to be added.)*

## Recommendation
The default `sub_depth=2` (a skip every 2 layers) is the sweet spot; K ≤ 4 works, K ≥ 8 fails;
no normalization needed. Use `sub_depth=1` only to recover the legacy single-layer block.
````

For the results table use a fenced `{code-cell}` (myst-nb) with a missing-results guard:

```python
import json
from pathlib import Path
import pandas as pd

R = Path("../../benchmarks/results/deep-residual/trainability.json")
if not R.exists():
    print("No sweep results committed yet. Run "
          "`python -m benchmarks.deep_residual_run` (see the reproduce command above).")
else:
    df = pd.DataFrame(json.loads(R.read_text()))
    display(df.pivot_table(index=["mode", "depth"], columns="skip_k",
                           values="final_train_mse", aggfunc="first"))
```

- [ ] **Step 2: Wire into `docs/concepts/index.md`**

Read the file; add `monotonic-residual` to its `{toctree}` (and a one-line description bullet if the page lists sections). Match the file's existing structure.

- [ ] **Step 3: Verify docs build + pre-commit**

Run: `uv run pre-commit run --all-files --hook-stage manual && ./tools/build-docs.sh`
Expected: pre-commit clean (codespell/EOF — re-run until clean; reword rather than editing codespell config); Sphinx build succeeds with `monotonic-residual` in the concepts toctree and the missing-results guard printing the placeholder (results not committed until Task 7).

- [ ] **Step 4: Commit (unsigned)**

```bash
git add docs/concepts/monotonic-residual.md docs/concepts/index.md
git -c commit.gpgsign=false commit -m "docs(concepts): deep monotonic residual — theory + methods (paper-grade)"
```

---

### Task 7: Controller phase — run sweep, render, sign, PR

> **Not a subagent TDD task.** The controller runs this after Tasks 1–6 pass review and the whole-branch review is clean.

- [ ] **Step 1: Run the sweep**

```bash
OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 uv run --extra torch --group bench python -m benchmarks.deep_residual_run
```
Confirm the committed `benchmarks/results/deep-residual/trainability.json` shows K∈{1,2,4} training deep (MSE ~0.07–0.11) and plain/K=8 failing — matching the design's §2.

- [ ] **Step 2: Re-render the docs table**

```bash
uv run --group bench --group docs --extra torch jupyter nbconvert --to notebook --execute --inplace docs/concepts/monotonic-residual.md
```
(If the page is `.md` with a myst-nb `{code-cell}`, execution happens at build; otherwise render the notebook form. Verify the table populates.)

- [ ] **Step 3: Commit results (unsigned)**

```bash
git add benchmarks/results/deep-residual/trainability.json docs/concepts/monotonic-residual.md
git -c commit.gpgsign=false commit -m "bench(deep-residual): committed skip-K sweep results + rendered table"
```

- [ ] **Step 4: Re-sign, push, PR**

```bash
git rebase --exec "git commit --amend --no-edit -n -S" $(git merge-base main HEAD)
git log --format="%h %G? %s" $(git merge-base main HEAD)..HEAD   # expect all G
git push -u origin feat/deep-mono-residual
gh pr create --title "Deep monotonic networks via MonoResidual sub_depth" --body-file <notes>
```

- [ ] **Step 5: Confirm CI green**

`gh pr checks <n>` — all test legs + static-analysis + pre-commit + docs-smoke pass.

---

## Notes for the executor

- Run mypy as `uv run --group bench mypy` (canonical gate; benchmarks/ imports need bench).
- Backend tasks: matching `MONONET_TEST_BACKEND` + `pytest.importorskip`.
- Default is now `sub_depth=2` (sentinel `None`→2; default `F` is a 2-layer `Sequential`). `sub_depth=1` must reproduce the LEGACY single-`MonoLinear`/`MonoDense` `F` (NOT wrapped in `Sequential`) — Tasks 1–3 branch on `k == 1` for exactly this. `F` alone (no `sub_depth`) must NOT raise (sentinel makes this work); `F` + explicit `sub_depth` raises.
- Monotonicity tests use loose fp tolerances (torch f64 `1e-9`; jax `1e-4`; keras f32 `1e-3`) — do not tighten.
- Stage 2 (real-dataset accuracy) is a documented follow-on, not in this plan.
- **Default change accepted to flow into benchmarks (decision (b)).** `model_builder.py` builds
  `MonoResidual` without `sub_depth`, so the Phase-2a "residual" flavor now becomes 2-deep. We do
  NOT pin `sub_depth=1` there. Consequence: the committed `docs/benchmarks/flavor-comparison`
  residual-flavor numbers are now **stale** — re-running Phase-2a under the new default is a
  tracked **follow-up**, out of this plan's scope. (The `deep_residual_run` sweep in Task 4 passes
  explicit `sub_depth`, so it is unaffected.)
- **Cross-backend parity** (spec §6): no dedicated task. `sub_depth` adds *no new kernel* — it
  only composes the already-equivalence-tested `monotonic_dense` via each framework's
  `Sequential`, so numerical parity is inherited from the unchanged stateless-kernel equivalence
  harness, and behaviour is pinned per-backend by the Task 1–3 monotonicity tests. A dedicated
  composed-block parity test (weight injection across frameworks) is deferred as YAGNI.
