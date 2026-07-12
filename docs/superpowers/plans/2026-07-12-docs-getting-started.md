# Getting-started Quickstart + Consolidated Tabbed Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal backend-tabbed quickstart + persona routing to the docs landing page, and consolidate the three per-backend guides into one tabbed guide (shared prose once, backend bits in tabs).

**Architecture:** Documentation work. New minimal example modules under `docs/examples/` are `literalinclude`-d into `sphinx-design` `{tab-set}`s and guarded by smoke tests, so the pages can't drift from runnable code. Gates are the strict Sphinx build (`sphinx-build -W`) and `pytest tests/examples`.

**Tech Stack:** Sphinx + myst-nb, `sphinx-design` tabs (enabled in `docs/conf.py`), MyST `colon_fence` (enabled), `literalinclude`, pytest.

**Spec:** [docs/superpowers/specs/2026-07-12-docs-getting-started-design.md](../specs/2026-07-12-docs-getting-started-design.md)

## Global Constraints

- **Branch:** `spec/docs-getting-started` (already checked out). Never commit to `main`.
- **Commit signing is broken in this container** — always `git commit --no-gpg-sign`.
- **Deferred:** #3 upgrade note is NOT in this plan (release-time work).
- `mononet` ships **layers, not composed models** — quickstarts stack them with the native `Sequential`; there is no `MonoMLP`.
- Dense layer names: **`MonoLinear`** for PyTorch/JAX, **`MonoDense`** for Keras. Shared across all: `MonoResidual`, `MonoInput`. torch/jax dense layers take `(in_features, units, ...)`; jax layers require `rngs=nnx.Rngs(...)`; keras `MonoDense(units, ...)` infers input width at build.
- Default `mode="absolute"`; examples pass `activation="elu"` explicitly (default is `identity`).
- Never run any `uv sync` variant (prunes backend extras); the env already has all three backends. Run tools/tests via `uv run`. Keras needs `KERAS_BACKEND=jax`.
- **Validation gates (must pass):** `./tools/build-docs.sh` (strict `-W`); `uv run pytest tests/examples`.
- Tab syntax: MyST colon-fence nesting — `::::{tab-set}` / `:::{tab-item} <label>` / inner ` ```{literalinclude} ``` `.
- `literalinclude` paths are relative to the including file: from `docs/index.md` → `examples/<file>`; from `docs/guides/index.md` → `../examples/<file>`.

---

### Task 1: Minimal quickstart example modules + smoke tests

**Files:**
- Create: `docs/examples/quickstart_torch.py`, `docs/examples/quickstart_jax.py`, `docs/examples/quickstart_keras.py`
- Create: `tests/examples/test_quickstart.py`

**Interfaces:**
- Consumes: `mononet.torch.{MonoLinear,MonoResidual}`, `mononet.jax.{MonoLinear,MonoResidual}`, `mononet.keras.{MonoDense,MonoResidual}`; `tests/examples/_loader.load_example(filename) -> module`.
- Produces: three example modules that each build a model and, at module level, assign `y = model(<toy batch>)` of shape `(8, 1)` and `print` its shape. Task 2 `literalinclude`s these three files.

- [ ] **Step 1: Write `tests/examples/test_quickstart.py` (failing — modules don't exist yet)**

```python
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests: each landing-page quickstart module builds and runs."""

from __future__ import annotations

import os

import pytest

from tests.examples._loader import load_example


def test_quickstart_torch() -> None:
    pytest.importorskip("torch")
    mod = load_example("quickstart_torch.py")
    assert tuple(mod.y.shape) == (8, 1)


def test_quickstart_jax() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("flax.nnx")
    mod = load_example("quickstart_jax.py")
    assert tuple(mod.y.shape) == (8, 1)


def test_quickstart_keras() -> None:
    os.environ.setdefault("KERAS_BACKEND", "jax")
    pytest.importorskip("keras")
    mod = load_example("quickstart_keras.py")
    assert tuple(mod.y.shape) == (8, 1)
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `KERAS_BACKEND=jax uv run pytest tests/examples/test_quickstart.py -q`
Expected: FAIL — `load_example` raises `FileNotFoundError`/import error because the quickstart modules don't exist yet.

- [ ] **Step 3: Write `docs/examples/quickstart_torch.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Quickstart: a monotone regressor in PyTorch.

Non-decreasing in every one of its 4 inputs. ``mononet`` ships layers, not
composed models — stack them with a native ``torch.nn.Sequential``.
"""

from __future__ import annotations

import torch
from torch import nn

from mononet.torch import MonoLinear, MonoResidual

model = nn.Sequential(
    MonoLinear(4, 32, activation="elu"),
    MonoResidual(32, 32, activation="elu"),
    MonoLinear(32, 1),
)

y = model(torch.rand(8, 4))
print(y.shape)  # torch.Size([8, 1]) — monotone in all 4 inputs
```

- [ ] **Step 4: Write `docs/examples/quickstart_jax.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Quickstart: a monotone regressor in JAX / Flax NNX.

Non-decreasing in every one of its 4 inputs. Dense layers take an explicit
``rngs`` for weight initialization.
"""

from __future__ import annotations

import jax
from flax import nnx

from mononet.jax import MonoLinear, MonoResidual

rngs = nnx.Rngs(0)
model = nnx.Sequential(
    MonoLinear(4, 32, activation="elu", rngs=rngs),
    MonoResidual(32, 32, activation="elu", rngs=rngs),
    MonoLinear(32, 1, rngs=rngs),
)

y = model(jax.random.uniform(jax.random.key(0), (8, 4)))
print(y.shape)  # (8, 1) — monotone in all 4 inputs
```

- [ ] **Step 5: Write `docs/examples/quickstart_keras.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Quickstart: a monotone regressor in Keras 3.

Non-decreasing in every one of its 4 inputs. Runs on whichever backend Keras is
configured to use (``KERAS_BACKEND``); ``MonoDense`` infers the input width.
"""

from __future__ import annotations

import keras

from mononet.keras import MonoDense, MonoResidual

model = keras.Sequential(
    [
        MonoDense(32, activation="elu"),
        MonoResidual(32, activation="elu"),
        MonoDense(1),
    ]
)

y = model(keras.ops.zeros((8, 4)))
print(tuple(y.shape))  # (8, 1) — monotone in all 4 inputs
```

- [ ] **Step 6: Run the smoke tests to confirm they pass**

Run: `KERAS_BACKEND=jax uv run pytest tests/examples/test_quickstart.py -q`
Expected: PASS — `3 passed` (all three backends are installed in this env).

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format docs/examples/quickstart_torch.py docs/examples/quickstart_jax.py docs/examples/quickstart_keras.py tests/examples/test_quickstart.py
uv run ruff check --exit-non-zero-on-fix
git add docs/examples/quickstart_torch.py docs/examples/quickstart_jax.py docs/examples/quickstart_keras.py tests/examples/test_quickstart.py
git commit --no-gpg-sign -m "docs(examples): minimal per-backend quickstart modules + smoke tests"
```

---

### Task 2: Landing quickstart tabs + where-to-next routing

**Files:**
- Modify: `docs/index.md`

**Interfaces:**
- Consumes: the three `docs/examples/quickstart_*.py` files from Task 1 (via `literalinclude`).

- [ ] **Step 1: Insert the Quickstart + Where-to-next sections**

In `docs/index.md`, between the end of the `## Install` block (the paragraph ending "…the CPU-torch (uv vs pip) caveat.") and the `## Citation` heading, insert exactly:

````markdown
## Quickstart

Your first monotonic model — a small regressor that is **non-decreasing in every
input**. `mononet` ships layers; stack them with your framework's native
`Sequential`.

::::{tab-set}
:::{tab-item} PyTorch
```{literalinclude} examples/quickstart_torch.py
:language: python
```
:::
:::{tab-item} JAX
```{literalinclude} examples/quickstart_jax.py
:language: python
```
:::
:::{tab-item} Keras 3
```{literalinclude} examples/quickstart_keras.py
:language: python
```
:::
::::

The same layers exist in all three backends — see the [guide](guides/index.md)
for the full mixed-feature example.

## Where to next

- **Build something** — the [guides](guides/index.md): the full mixed-feature
  example and per-backend specifics.
- **Understand how it stays monotone** — [concepts](concepts/index.md).
- **See it work / reproduce results** — [benchmarks](benchmarks/index.md).
- **API details** — the [reference](reference.md).

````

(Leave `## Install`, `## Citation`, the BibTeX note, and the hidden `toctree` untouched. Resulting order: tagline → Install → Quickstart → Where to next → Citation → toctree.)

- [ ] **Step 2: Build the docs strictly and confirm the tabs render**

Run: `./tools/build-docs.sh`
Expected: `build succeeded` with zero warnings. Then confirm the three includes resolved:
Run: `grep -c "quickstart" docs/_build/html/index.html`
Expected: a non-zero count (the tab labels + included code are present in the rendered landing).

- [ ] **Step 3: Commit**

```bash
git add docs/index.md
git commit --no-gpg-sign -m "docs(index): tabbed quickstart + where-to-next routing"
```

---

### Task 3: Consolidate the three guides into one tabbed guide

**Files:**
- Modify: `docs/guides/index.md`
- Delete: `docs/guides/pytorch.md`, `docs/guides/jax.md`, `docs/guides/keras.md`

**Interfaces:**
- Consumes: the existing `docs/examples/risk_net_{torch,jax,keras}.py` (unchanged) via `literalinclude`.

- [ ] **Step 1: Replace `docs/guides/index.md` entirely**

Overwrite `docs/guides/index.md` with exactly:

````markdown
# Guides

`mononet` ships monotonic **layers**, not composed models — stack them with your
framework's native `Sequential` (or equivalent); there is no composed `MonoMLP`.
Every backend exposes the same three layers: the **dense monotone layer** (a
monotonic analogue of the framework's dense layer, non-decreasing in all
inputs), **`MonoResidual`** (a dual-gated monotone residual block, warm-started
near identity), and **`MonoInput`** (a sign-flip layer encoding per-feature
monotonicity directions).

## Example

A mixed-feature network: monotone in 3 features (2 non-decreasing, 1
non-increasing) via `MonoInput`, and unconstrained in 2 non-monotone features,
which are embedded through a plain MLP before being concatenated with the
monotone path. The embedding absorbs the non-monotonicity, so the composite
`RiskNet` is monotone in `x_mono` and free in `x_free`. The dense layer and
`MonoResidual` default to `mode="absolute"`. Pick your backend:

::::{tab-set}
:::{tab-item} PyTorch
`mononet.torch` provides monotonic layers as {py:class}`torch.nn.Module`
subclasses; they drop into any training loop (plain PyTorch, PyTorch Lightning,
…) and compose with {py:class}`torch.nn.Sequential`.

    pip install "mononet[torch]"

Layers: {py:class}`mononet.torch.layers.MonoLinear` (monotonic
{py:class}`torch.nn.Linear`), {py:class}`mononet.torch.layers.MonoResidual`,
{py:class}`mononet.torch.layers.MonoInput`.

```{literalinclude} ../examples/risk_net_torch.py
:language: python
```

For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of `{-1, +1}`) to
`MonoInput`.
:::
:::{tab-item} JAX
`mononet.jax` uses **Flax NNX** — layers are `flax.nnx.Module` subclasses, fully
compatible with {py:func}`jax.jit` and {py:func}`jax.grad`, and compose with
`flax.nnx.Sequential`.

    pip install "mononet[jax]"

Layers: {py:class}`mononet.jax.layers.MonoLinear` (monotonic `flax.nnx.Linear`),
{py:class}`mononet.jax.layers.MonoResidual`,
{py:class}`mononet.jax.layers.MonoInput`.

```{literalinclude} ../examples/risk_net_jax.py
:language: python
```

The dense layers take an explicit `rngs` ({py:class}`flax.nnx.Rngs`) for weight
initialization. For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of `{-1, +1}`) to
`MonoInput`.
:::
:::{tab-item} Keras 3
`mononet.keras` uses `keras.ops`, so the same code runs whether Keras is
configured to use JAX, TensorFlow, or PyTorch under the hood (the GPU
devcontainer ships with `KERAS_BACKEND=jax`).

    pip install "mononet[keras]"

Layers: {py:class}`mononet.keras.layers.MonoDense` (monotonic
`keras.layers.Dense`), {py:class}`mononet.keras.layers.MonoResidual`,
{py:class}`mononet.keras.layers.MonoInput`.

```{literalinclude} ../examples/risk_net_keras.py
:language: python
```

`MonoDense` infers the input width at build time (Keras style) — no
`in_features`. `MonoDense` and `MonoInput` implement `get_config`/`from_config`,
so models serialize with the standard Keras saving APIs. For per-feature
monotonicity directions, pass a {py:class}`~mononet.core.types.MonotonicityMask`
(a 1-D array of `{-1, +1}`) to `MonoInput`.
:::
::::

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
````

Note: this removes the child `{toctree}` that previously listed `pytorch`/`jax`/`keras`. The top-level `docs/index.md` toctree entry `guides/index` remains and now points at this single leaf page.

- [ ] **Step 2: Delete the three per-backend guide files**

```bash
git rm docs/guides/pytorch.md docs/guides/jax.md docs/guides/keras.md
```

- [ ] **Step 3: Strict build — confirms no dangling refs/orphans from the deletion**

Run: `./tools/build-docs.sh`
Expected: `build succeeded`, zero warnings. A dangling link to the deleted pages, an orphaned page, or a broken tab/xref would fail the `-W` build here.

- [ ] **Step 4: Re-run the link/nitpicky check for new internal breaks**

Run: `./tools/check-docs.sh`
Expected: no *new* broken internal links or dangling `doc` references versus the pre-existing catalogue (the known external/flaky failures from the audit — pytorch-docs anchors, justia 403 — may remain; no new `guides/pytorch`-style breaks should appear).

- [ ] **Step 5: Commit**

```bash
git add docs/guides/index.md
git commit --no-gpg-sign -m "docs(guides): consolidate per-backend guides into one tabbed guide"
```

---

## After all tasks

Do NOT open the PR from within a task — the finishing step
(superpowers:finishing-a-development-branch, after the whole-branch review)
handles push/PR. The PR should note that finding #3 (upgrade note) was
deferred to release time, and that the audit report's follow-up list should be
updated (`#2` resolved; guide-duplication resolved) when this merges.

## Notes for the implementer

- `docs/_build/` is a build artifact — never `git add` it.
- If the strict build reports a tab-directive parse error, check the colon-fence
  nesting depth: the outer `tab-set` uses four colons (`::::`), each `tab-item`
  three (`:::`), and the inner `literalinclude` a normal triple-backtick fence.
- The quickstart smoke tests and the existing `risk_net_*`/README parity tests
  run in the same `tests/examples` suite; run the whole suite once before the
  final validation: `KERAS_BACKEND=jax uv run pytest tests/examples -q`.
