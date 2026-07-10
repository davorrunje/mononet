# Default mode → absolute + Mode literal + tested examples — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `absolute` the default `mode`, type the layer `mode` param as the `Mode` literal, and replace the docs examples with a tested, coverage-included mixed-feature `RiskNet`.

**Architecture:** Default-value + type-annotation changes in the three backend layer modules and the two config dataclasses + NumPy reference; a small `Mode` propagation into benchmark helpers; and new per-backend example modules under `docs/examples/` that the guides `literalinclude`, the README embeds inline (drift-guarded), and `tests/examples/` exercises for monotonicity.

**Tech Stack:** Python 3.11+, PyTorch, Flax NNX (JAX), Keras 3, stdlib dataclasses, pytest, pytest-cov, uv, Sphinx/myst-nb.

## Global Constraints

- Breaking change is acceptable (`0.0.0a0`); add a `CHANGELOG` recovery note.
- Do NOT touch: kernels (`_kernels.py`), the equivalence harness (`tests/equivalence/`), `convex_fraction`, the residual gates, `MonoInput`/`MonotonicityMask` semantics, or the #75 activation contract.
- `Mode = Literal["switch", "absolute"]` already exists in `mononet/core/config.py`; `ActivationName` in `mononet/core/types.py`.
- Line length 88 (ruff); strict mypy (`files = mononet, tests, benchmarks` — `docs/` is NOT type-checked). ruff lints `docs/**/*.py` but D100–D107 are per-file-ignored there.
- Branch: `feat/mode-default-absolute` (already checked out).
- Environment: gpu-torch container — torch tests run; JAX/Keras tests `importorskip`-SKIP here (expected); `uv run mypy` checks all backends' source. Full JAX/Keras runtime verification happens in CI / the CPU `default` container.
- Commit after each task. Do NOT use `--no-verify`; if the pre-commit `docs` hook fails purely on a locale error, prefix the commit with `LC_ALL=C.UTF-8 LANG=C.UTF-8`.

---

### Task 1: Default `mode` → `absolute` (source + config test + CHANGELOG)

**Files:**
- Modify: `mononet/core/config.py` (`MonoConfig.mode` ~line 24, `MonoResidualConfig.mode` ~line 89)
- Modify: `mononet/core/reference.py` (`monotonic_dense` `mode` default ~line 127)
- Modify: `mononet/torch/layers.py` (MonoLinear ~74, MonoResidual ~139), `mononet/jax/layers.py` (~82, ~159), `mononet/keras/layers.py` (~66, ~175)
- Modify: `tests/core/test_config.py`, `CHANGELOG.md`
- Test: `tests/core/test_config.py`, `tests/torch/test_default_mode.py` (new)

**Interfaces:**
- Produces: default `mode` is `"absolute"` everywhere; `MonoConfig().mode == "absolute"`; a bare `MonoLinear`/`MonoDense`/`MonoResidual` (no `mode=`) builds in absolute mode.

- [ ] **Step 1: Update/adjust the failing tests**

In `tests/core/test_config.py`, change the defaults assertion (line ~26) from `assert cfg.mode == "switch"` to:

```python
    assert cfg.mode == "absolute"
```

Create `tests/torch/test_default_mode.py`:

```python
from __future__ import annotations

import pytest

pytest.importorskip("torch")

from mononet.torch import MonoLinear, MonoResidual  # noqa: E402


def test_monolinear_default_mode_is_absolute() -> None:
    """A bare MonoLinear defaults to absolute mode."""
    assert MonoLinear(4, 8).mode == "absolute"


def test_monoresidual_default_mode_is_absolute() -> None:
    """A bare MonoResidual (default F) builds its sublayers in absolute mode."""
    block = MonoResidual(8, 8, activation="elu")
    sub = block.F[0] if hasattr(block.F, "__getitem__") else block.F
    assert sub.mode == "absolute"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/core/test_config.py::test_mono_config_defaults tests/torch/test_default_mode.py -v`
Expected: FAIL — defaults are still `"switch"`.

- [ ] **Step 3: Flip the defaults**

Change each occurrence of the default `mode` value from `"switch"` to `"absolute"`:
- `mononet/core/config.py`: `mode: Mode = "switch"` → `mode: Mode = "absolute"` (both `MonoConfig` and `MonoResidualConfig`).
- `mononet/core/reference.py`: `mode: str = "switch"` → `mode: str = "absolute"` (`monotonic_dense`).
- `mononet/torch/layers.py`, `mononet/jax/layers.py`, `mononet/keras/layers.py`: every `mode: str = "switch"` → `mode: str = "absolute"` (MonoLinear/MonoDense and MonoResidual — 2 per file).

Update the `:param mode:` docstrings that say `"switch" (default)` to note `"absolute"` is the default (torch/jax/keras leaf + residual, config, reference).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/core/test_config.py tests/torch/test_default_mode.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — full torch + core + equivalence + benchmarks**

Run: `uv run pytest tests/torch tests/core tests/equivalence tests/benchmarks -q`
Expected: PASS/skip. (Equivalence vectors and other tests set `mode` explicitly, so they are unaffected.) If any test that omitted `mode` and asserted switch-specific output now fails, add an explicit `mode="switch"` to that construction to preserve its original intent, and note it in the commit.

- [ ] **Step 6: CHANGELOG**

Under `## [Unreleased]` → `### Changed` in `CHANGELOG.md`, add:

```markdown
- **BREAKING:** the default `mode` is now `"absolute"` (was `"switch"`) for
  `MonoLinear` / `MonoDense` / `MonoResidual` / `MonoConfig` /
  `MonoResidualConfig`. `absolute` uses the static init (no `init` needed) and
  is the paper's base `|W|` construction — pass `mode="switch"` explicitly to
  keep the previous behaviour.
```

- [ ] **Step 7: Lint + type-check + commit**

Run: `uv run ruff check mononet tests && uv run mypy`
Expected: clean.

```bash
git add mononet/core/config.py mononet/core/reference.py \
        mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py \
        tests/core/test_config.py tests/torch/test_default_mode.py CHANGELOG.md
git commit -m "feat: default mode to absolute across layers, configs, reference"
```

---

### Task 2: Type the layer `mode` param as `Mode`

**Files:**
- Modify: `mononet/torch/layers.py`, `mononet/jax/layers.py`, `mononet/keras/layers.py` (MonoLinear/MonoDense + MonoResidual `mode` param + a `TYPE_CHECKING` import)
- Modify: `benchmarks/_common/init_diagnostics.py` (`_stack`, `grad_flow`, `trainability`), `benchmarks/deep_init_run.py`

**Interfaces:**
- Consumes: `Mode` from `mononet.core.config`.
- Produces: all public layer `mode` params typed `Mode`; benchmark helper `mode` params typed `Mode`.

- [ ] **Step 1: Tighten the layer `mode` params (all 3 backends)**

In each of `mononet/torch/layers.py`, `mononet/jax/layers.py`, `mononet/keras/layers.py`: change every `mode: str = "absolute",` (MonoLinear/MonoDense and MonoResidual) to `mode: Mode = "absolute",`. Add the import under the existing `if TYPE_CHECKING:` block (create the block if absent, as in `mononet/keras/layers.py`):

```python
    from mononet.core.config import Mode
```

(torch and jax already have a `TYPE_CHECKING` block with `from collections.abc import Callable` — add the `Mode` line there. Keras: add `if TYPE_CHECKING:` after the imports.)

- [ ] **Step 2: Propagate `Mode` to benchmark helpers**

In `benchmarks/_common/init_diagnostics.py`: import `Mode` under `TYPE_CHECKING` (`from mononet.core.config import Mode`) and change the `mode: str` params of `_stack`, `grad_flow`, and `trainability` to `mode: Mode`.

In `benchmarks/deep_init_run.py`: find any `mode: str` parameter or local passed into a layer/helper and type it `Mode` (import under `TYPE_CHECKING`). If `mode` is read from an untyped source (e.g. argparse/JSON), `cast("Mode", value)` at that boundary — mirror the `ActivationName` cast added in `benchmarks/_common/config_io.py` in PR #75.

- [ ] **Step 3: Type-check (the gate for this task)**

Run: `uv run mypy`
Expected: `Success` across all source files. If new `arg-type` errors appear at other benchmark/test call sites passing a bare `str` `mode`, type that source as `Mode` or `cast("Mode", …)` at the boundary until clean. Then `uv run ruff check mononet benchmarks` (ruff `--fix` will relocate any `TYPE_CHECKING`-only import).

- [ ] **Step 4: Sanity test + commit**

Run: `uv run pytest tests/torch tests/core -q`
Expected: PASS/skip (no behavioural change).

```bash
git add mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py \
        benchmarks/_common/init_diagnostics.py benchmarks/deep_init_run.py
git commit -m "refactor: type layer mode param as Mode literal; propagate to benchmarks"
```

---

### Task 3: PyTorch example module + monotonicity test + coverage

**Files:**
- Create: `docs/examples/risk_net_torch.py`
- Create: `tests/examples/__init__.py`, `tests/examples/_loader.py`, `tests/examples/test_risk_net_torch.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options].addopts`)

**Interfaces:**
- Produces: `docs/examples/risk_net_torch.py` defines `class RiskNet(nn.Module)` with `forward(self, x_mono, x_free)`; `tests/examples/_loader.py` exposes `load_example(filename) -> module`.

- [ ] **Step 1: Write the failing test + loader**

Create `tests/examples/__init__.py` (empty).

Create `tests/examples/_loader.py`:

```python
"""Load a docs/examples/*.py module by file path (kept out of the package)."""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "docs" / "examples"


def load_example(filename: str) -> ModuleType:
    """Import a ``docs/examples`` module from its file path.

    :param filename: File name under ``docs/examples`` (e.g. ``risk_net_torch.py``).
    :returns: The imported module object.
    """
    path = _EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
```

Create `tests/examples/test_risk_net_torch.py`:

```python
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.examples._loader import load_example  # noqa: E402


def test_risk_net_forward_and_monotone() -> None:
    """RiskNet runs and is monotone in x_mono (dirs +1, +1, -1), free in x_free."""
    mod = load_example("risk_net_torch.py")
    torch.manual_seed(0)
    net = mod.RiskNet()
    x_mono = torch.randn(16, 3)
    x_free = torch.randn(16, 2)
    y = net(x_mono, x_free)
    assert tuple(y.shape) == (16, 1)
    with torch.no_grad():
        base = net(x_mono, x_free)
        for j, sign in ((0, 1), (1, 1), (2, -1)):
            bumped = x_mono.clone()
            bumped[:, j] += 0.5
            diff = (net(bumped, x_free) - base).squeeze(-1)
            if sign > 0:
                assert bool((diff >= -1e-4).all())
            else:
                assert bool((diff <= 1e-4).all())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/examples/test_risk_net_torch.py -v`
Expected: FAIL — `docs/examples/risk_net_torch.py` does not exist.

- [ ] **Step 3: Write the example module**

Create `docs/examples/risk_net_torch.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Mixed-feature monotone network (PyTorch).

Monotone in 3 features (2 non-decreasing, 1 non-increasing) via ``MonoInput``,
and unconstrained in 2 non-monotone features, which are embedded through a
plain MLP. The embedding absorbs the non-monotonicity, so the composite map is
monotone in ``x_mono`` and free in ``x_free``. Absolute mode is the default.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mononet import MonotonicityMask
from mononet.torch import MonoInput, MonoLinear, MonoResidual


class RiskNet(nn.Module):
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.net = nn.Sequential(
            MonoLinear(11, 64, activation="elu"),
            MonoResidual(64, 64, activation="elu"),
            MonoResidual(64, 64, activation="elu"),
            MonoLinear(64, 1),
        )

    def forward(self, x_mono: torch.Tensor, x_free: torch.Tensor) -> torch.Tensor:
        """Combine the sign-flipped monotone features with the free embedding."""
        z = torch.cat([self.mono_in(x_mono), self.embed(x_free)], dim=-1)
        return self.net(z)
```

- [ ] **Step 4: Add `docs/examples` to coverage**

In `pyproject.toml`, `[tool.pytest.ini_options].addopts` (~line 149), append `--cov=docs/examples`:

```toml
addopts = '--cov=mononet --cov=benchmarks --cov=docs/examples --cov-append --cov-branch --cov-report=term-missing'
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/examples/test_risk_net_torch.py -v`
Expected: PASS (forward + all three monotonicity checks).

- [ ] **Step 6: Lint + commit**

Run: `uv run ruff check docs/examples tests/examples`
Expected: clean.

```bash
git add docs/examples/risk_net_torch.py tests/examples/__init__.py \
        tests/examples/_loader.py tests/examples/test_risk_net_torch.py pyproject.toml
git commit -m "docs+test: mixed-feature RiskNet example (torch) with monotonicity test"
```

---

### Task 4: JAX + Keras example modules + tests

**Files:**
- Create: `docs/examples/risk_net_jax.py`, `docs/examples/risk_net_keras.py`
- Create: `tests/examples/test_risk_net_jax.py`, `tests/examples/test_risk_net_keras.py`

**Interfaces:**
- Consumes: `tests/examples/_loader.load_example` (Task 3).
- Produces: `RiskNet` in each module — JAX `RiskNet(*, rngs)` with `__call__(x_mono, x_free)`; Keras `RiskNet()` with `call(x_mono, x_free)`.

- [ ] **Step 1: Write the failing tests**

`tests/examples/test_risk_net_jax.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from tests.examples._loader import load_example  # noqa: E402


def test_risk_net_forward_and_monotone() -> None:
    """JAX RiskNet runs and is monotone in x_mono (+1, +1, -1)."""
    mod = load_example("risk_net_jax.py")
    net = mod.RiskNet(rngs=nnx.Rngs(0))
    rng = np.random.default_rng(0)
    x_mono = jnp.asarray(rng.standard_normal((16, 3)), dtype=jnp.float32)
    x_free = jnp.asarray(rng.standard_normal((16, 2)), dtype=jnp.float32)
    base = np.asarray(net(x_mono, x_free))
    assert base.shape == (16, 1)
    for j, sign in ((0, 1), (1, 1), (2, -1)):
        bumped = x_mono.at[:, j].add(0.5)
        diff = (np.asarray(net(bumped, x_free)) - base)[:, 0]
        assert (diff >= -1e-4).all() if sign > 0 else (diff <= 1e-4).all()
```

`tests/examples/test_risk_net_keras.py`:

```python
import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("keras")

from tests.examples._loader import load_example  # noqa: E402


def test_risk_net_forward_and_monotone() -> None:
    """Keras RiskNet runs and is monotone in x_mono (+1, +1, -1)."""
    mod = load_example("risk_net_keras.py")
    net = mod.RiskNet()
    rng = np.random.default_rng(0)
    x_mono = rng.standard_normal((16, 3)).astype("float32")
    x_free = rng.standard_normal((16, 2)).astype("float32")
    base = np.asarray(net(x_mono, x_free))
    assert base.shape == (16, 1)
    for j, sign in ((0, 1), (1, 1), (2, -1)):
        bumped = x_mono.copy()
        bumped[:, j] += 0.5
        diff = (np.asarray(net(bumped, x_free)) - base)[:, 0]
        assert (diff >= -1e-4).all() if sign > 0 else (diff <= 1e-4).all()
```

- [ ] **Step 2: Run to verify they fail (or skip in torch-only env)**

Run: `uv run pytest tests/examples/test_risk_net_jax.py tests/examples/test_risk_net_keras.py -v`
Expected: in the gpu-torch container these SKIP (jax/keras not installed). In a jax/keras env they FAIL (modules missing). Either way, proceed to write the modules; CI verifies.

- [ ] **Step 3: Write `docs/examples/risk_net_jax.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Mixed-feature monotone network (JAX / Flax NNX). See risk_net_torch.py."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from flax import nnx

from mononet import MonotonicityMask
from mononet.jax import MonoInput, MonoLinear, MonoResidual


class RiskNet(nnx.Module):
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.embed1 = nnx.Linear(2, 16, rngs=rngs)
        self.embed2 = nnx.Linear(16, 8, rngs=rngs)
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.l1 = MonoLinear(11, 64, activation="elu", rngs=rngs)
        self.r1 = MonoResidual(64, 64, activation="elu", rngs=rngs)
        self.r2 = MonoResidual(64, 64, activation="elu", rngs=rngs)
        self.head = MonoLinear(64, 1, rngs=rngs)

    def __call__(self, x_mono: jnp.ndarray, x_free: jnp.ndarray) -> jnp.ndarray:
        """Combine the sign-flipped monotone features with the free embedding."""
        h = nnx.relu(self.embed1(x_free))
        h = nnx.relu(self.embed2(h))
        z = jnp.concatenate([self.mono_in(x_mono), h], axis=-1)
        return self.head(self.r2(self.r1(self.l1(z))))
```

- [ ] **Step 4: Write `docs/examples/risk_net_keras.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Mixed-feature monotone network (Keras 3). See risk_net_torch.py."""

from __future__ import annotations

from typing import Any

import keras
import numpy as np

from mononet import MonotonicityMask
from mononet.keras import MonoDense, MonoInput, MonoResidual


class RiskNet(keras.Model):  # type: ignore[misc]
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed1 = keras.layers.Dense(16, activation="relu")
        self.embed2 = keras.layers.Dense(8, activation="relu")
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.l1 = MonoDense(64, activation="elu")
        self.r1 = MonoResidual(64, activation="elu")
        self.r2 = MonoResidual(64, activation="elu")
        self.head = MonoDense(1)

    def call(self, x_mono: Any, x_free: Any) -> Any:
        """Combine the sign-flipped monotone features with the free embedding."""
        h = self.embed2(self.embed1(x_free))
        z = keras.ops.concatenate([self.mono_in(x_mono), h], axis=-1)
        return self.head(self.r2(self.r1(self.l1(z))))
```

- [ ] **Step 5: Run tests (skip-clean here) + ruff**

Run: `uv run pytest tests/examples -v` (torch passes; jax/keras skip here)
Run: `uv run ruff check docs/examples tests/examples`
Expected: ruff clean; torch test passes; jax/keras skip. Flag in the report that jax/keras runtime verification is pending CI / default container.

- [ ] **Step 6: Commit**

```bash
git add docs/examples/risk_net_jax.py docs/examples/risk_net_keras.py \
        tests/examples/test_risk_net_jax.py tests/examples/test_risk_net_keras.py
git commit -m "docs+test: RiskNet example (jax, keras) with monotonicity tests"
```

---

### Task 5: Wire examples into the guides + README (drift-guarded)

**Files:**
- Modify: `docs/guides/pytorch.md`, `docs/guides/jax.md`, `docs/guides/keras.md`, `README.md`
- Create: `tests/examples/test_readme_matches.py`

**Interfaces:**
- Consumes: the example modules (Tasks 3–4).

- [ ] **Step 1: Replace the guide code blocks with `literalinclude`**

In `docs/guides/pytorch.md`, replace the existing ```` ```python … ``` ```` quick-start block with:

````markdown
```{literalinclude} ../examples/risk_net_torch.py
:language: python
```
````

Do the same in `docs/guides/jax.md` (`../examples/risk_net_jax.py`) and `docs/guides/keras.md` (`../examples/risk_net_keras.py`). Keep/adjust the surrounding prose to explain the mixed-feature construction (monotone features via `MonoInput`; non-monotone features embedded through a plain MLP; absolute mode default).

- [ ] **Step 2: Embed the torch example inline in the README**

In `README.md`, replace the quick-start ```` ```python … ``` ```` block (lines ~35–48) with a python block whose body is byte-identical to `docs/examples/risk_net_torch.py` **minus the first `# SPDX…` line** (i.e. starting at the module docstring). Keep the surrounding "Quick start" prose.

- [ ] **Step 3: Write the README drift-guard test**

Create `tests/examples/test_readme_matches.py`:

```python
from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_readme_torch_block_matches_example_module() -> None:
    """README's torch example must equal docs/examples/risk_net_torch.py (sans SPDX)."""
    module_src = (_ROOT / "docs/examples/risk_net_torch.py").read_text()
    body = module_src.split("\n", 1)[1] if module_src.startswith("# SPDX") else module_src
    readme = (_ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    assert any(block.strip() == body.strip() for block in blocks), (
        "README torch example drifted from docs/examples/risk_net_torch.py"
    )
```

- [ ] **Step 4: Run drift-guard + docs build**

Run: `uv run pytest tests/examples/test_readme_matches.py -v`
Expected: PASS.
Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html`
Expected: `build succeeded`, no warnings (the `literalinclude` paths resolve and the modules import).

- [ ] **Step 5: Full pre-commit gate + commit**

Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run pre-commit run --all-files`
Expected: all hooks Passed/Skipped.

```bash
git add docs/guides/pytorch.md docs/guides/jax.md docs/guides/keras.md \
        README.md tests/examples/test_readme_matches.py
git commit -m "docs: use RiskNet example in guides (literalinclude) + README (drift-guarded)"
```

---

## Self-Review

**Spec coverage:**
- §3a default mode → absolute (configs + reference + 3 backends) + CHANGELOG + default test → Task 1. ✓
- §3b `mode` → `Mode` literal + benchmark propagation → Task 2. ✓
- §3c example modules (torch/jax/keras) → Tasks 3–4; guides `literalinclude` + README inline + drift guard → Task 5; per-backend monotonicity tests → Tasks 3–4; coverage `--cov=docs/examples` → Task 3. ✓
- §5 migration (CHANGELOG recovery note) → Task 1 Step 6. ✓
- §6 testing (default-mode test, RiskNet forward+monotonicity, README match, mypy, strict docs build) → Tasks 1/3/4/5. ✓
- §7 out-of-scope untouched (kernels/equivalence/convex_fraction/gates/MonoInput+mask/activation) → enforced in Global Constraints. ✓

**Placeholder scan:** No TBD/TODO; every code step shows full code; every command has expected output.

**Type consistency:** `Mode` (from `mononet.core.config`) used identically in Task 2 across backends + benchmark helpers. `RiskNet` signatures match between the example modules (Tasks 3–4) and the tests that construct them: torch `RiskNet()` / `forward(x_mono, x_free)`; jax `RiskNet(rngs=…)` / `__call__(x_mono, x_free)`; keras `RiskNet()` / `call(x_mono, x_free)`. `load_example(filename)` defined in Task 3, consumed in Tasks 3–4. Coverage `addopts` string in Task 3 matches the current value plus `--cov=docs/examples`.
