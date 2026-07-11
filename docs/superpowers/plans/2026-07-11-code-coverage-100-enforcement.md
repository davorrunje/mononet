# 100% Code-Coverage Enforcement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reach 100% test coverage on `mononet/` + `docs/examples/`, and enforce it on every PR via a hard CI gate plus Codecov status checks.

**Architecture:** A dedicated ubuntu CI job installs all CPU backends and runs the whole suite in one process, appending coverage across three backend passes into one data file — the single source of truth. `coverage report --fail-under=100` is the deterministic gate; Codecov `project`/`patch` statuses add per-PR diff annotation. The existing cross-OS/Python matrix keeps running for correctness with coverage disabled. The gap is closed by writing real tests (no `# pragma: no cover`).

**Tech Stack:** Python 3.11+, pytest + pytest-cov (coverage.py), uv, GitHub Actions, Codecov.

**Spec:** [docs/superpowers/specs/2026-07-11-code-coverage-100-enforcement-design.md](../specs/2026-07-11-code-coverage-100-enforcement-design.md)

## Global Constraints

- **Scope of 100%:** `mononet/` + `docs/examples/` only. `benchmarks/` is dropped from coverage entirely.
- **Never commit directly to `main`.** Work happens on branch `spec/coverage-100-enforcement` (already checked out).
- **This container signs commits with a macOS SSH key that is absent here** — commit with `git commit --no-gpg-sign`.
- Python 3.11+, ruff line length 88, strict mypy. Test files are exempt from docstring rules (`tests/**/*.py` per-file-ignores) and from `S101`/`PLR2004`/`N806`.
- **No `# pragma: no cover`** except for provably-unreachable defensive code, justified in the PR. The gap in this plan needs none.
- Backend-specific source is only covered under its own `MONONET_TEST_BACKEND`, so per-backend tests live in `tests/{torch,jax,keras}/`.
- Run tests through `uv run`. Keras uses the JAX backend: set `KERAS_BACKEND=jax`.

## Ground truth: the exact remaining gap

Measured on 2026-07-11 with the combined three-pass invocation (all backends, equivalence run under each). Total `mononet/`: **90%** — every uncovered line below is genuinely testable:

| File | Uncovered | What it is |
|---|---|---|
| `mononet/core/numerics.py` | 4–29 (whole module, 0%) | `default_atol` / `default_rtol` never imported by any test |
| `mononet/core/config.py` | 37, 98, 100 | `MonoConfig` bad-mode; `MonoResidualConfig` bad-units, bad-mode |
| `mononet/core/reference.py` | 43–45, 75, 117 | `base_activation` identity branch + unknown-name raise; `apply_gate` unknown-token raise; `monotonic_dense` unknown-mode raise |
| `mononet/core/init.py` | 87 | `_bisect` non-convergence fall-through (`return 0.5 * (lo + hi)`) |
| `mononet/torch/_kernels.py` | 40, 69, 104 | `activation` / `gate` / `monotonic_dense` error raises |
| `mononet/torch/layers.py` | 50, 52, 185, 224 | `_init_weight` InitSpec branch + seed branch; `MonoResidual` callable-`F` branch; `MonoInput` scalar-int branch |
| `mononet/jax/_kernels.py` | 30, 56, 91 | `activation` / `gate` / `monotonic_dense` error raises |
| `mononet/jax/layers.py` | 214, 253 | `MonoResidual` callable-`F` branch; `MonoInput` scalar-int branch |
| `mononet/keras/_kernels.py` | 30, 56, 93 | `activation` / `gate` / `monotonic_dense` error raises |
| `mononet/keras/layers.py` | 267–278, 301 | `MonoResidual.get_config`; `MonoInput` scalar-int branch |

---

### Task 1: Coverage config + CI scaffold + Codecov (gate NOT yet enforcing)

Lands all plumbing while leaving the build green (no `--fail-under` yet), per the spec's sequencing.

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options].addopts`; add `[tool.coverage.run]` + `[tool.coverage.report]`)
- Modify: `.github/workflows/build.yml` (add `coverage` job; add `--no-cov` to the matrix `test` job)
- Create: `codecov.yml`

**Interfaces:**
- Produces: a `coverage` CI job whose commands (`coverage erase` → three `pytest` passes → `coverage report` → `coverage xml`) are the exact local reproduction used by every later task's verification. Task 6 will add `--fail-under=100` to this job's `coverage report` and add the job to `check.needs`.

- [ ] **Step 1: Update `pyproject.toml` — drop benchmarks from cov scope, add coverage config**

Change the `addopts` line (currently line 172):

```toml
addopts = '--cov=mononet --cov=docs/examples --cov-append --cov-branch --cov-report=term-missing'
```

Then, immediately after the `[tool.pytest.ini_options]` block (before `[tool.ruff]`), add:

```toml
[tool.coverage.report]
exclude_also = [
    "if TYPE_CHECKING:",
    "\\.\\.\\.",
    "if __name__ == .__main__.:",
    "pragma: no cover",
]
```

(Run scope and branch mode stay defined by the `addopts` `--cov`/`--cov-branch`
flags; the bare `coverage report`/`coverage xml` commands in CI read the data
file those flags produce and apply the `[tool.coverage.report]` exclusions.)

- [ ] **Step 2: Add `--no-cov` to the matrix `test` job**

In `.github/workflows/build.yml`, the `test` job's run command (line 98) becomes:

```yaml
        run: pytest tests/core "tests/${{ matrix.backend }}" tests/equivalence tests/benchmarks tests/test_top_level_imports.py --no-cov -v
```

(Coverage is measured only by the new `coverage` job; the matrix stays for cross-OS/Python correctness.)

- [ ] **Step 3: Add the `coverage` job**

In `.github/workflows/build.yml`, add this job after the `test` job and before `check`:

```yaml
  coverage:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.13"
      - uses: astral-sh/setup-uv@v8.3.1
      - name: Install dependencies
        run: uv pip install --system -e ".[all-cpu]" --group=dev
      - name: Run combined-backend coverage
        env:
          KERAS_BACKEND: jax
        run: |
          coverage erase
          MONONET_TEST_BACKEND=torch pytest tests/core tests/torch tests/equivalence tests/examples tests/test_top_level_imports.py
          MONONET_TEST_BACKEND=jax   pytest tests/jax   tests/equivalence
          MONONET_TEST_BACKEND=keras pytest tests/keras tests/equivalence
          coverage report --show-missing
          coverage xml
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
          files: coverage.xml
          fail_ci_if_error: true
```

Note: `--fail-under=100` is deliberately absent here — it is added in Task 6. The `coverage` job is deliberately NOT yet in `check.needs` — also Task 6.

- [ ] **Step 4: Create `codecov.yml`**

```yaml
coverage:
  status:
    project:
      default:
        target: 100%
        threshold: 0%
    patch:
      default:
        target: 100%
comment:
  layout: "condensed_header, diff, files"
ignore:
  - "benchmarks/**"
  - "tests/**"
  - "docs/**"
```

- [ ] **Step 5: Verify the config locally (reproduce the CI job)**

Run:

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=torch uv run pytest tests/core tests/torch tests/equivalence tests/examples tests/test_top_level_imports.py -q
KERAS_BACKEND=jax MONONET_TEST_BACKEND=jax   uv run pytest tests/jax   tests/equivalence -q
KERAS_BACKEND=jax MONONET_TEST_BACKEND=keras uv run pytest tests/keras tests/equivalence -q
uv run coverage report
```

Expected: report lists only `mononet/*` and `docs/examples/*` (NO `benchmarks/*` rows), TOTAL near **90%**, and no `if TYPE_CHECKING:` bodies appear as missing.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format pyproject.toml
uv run ruff check --exit-non-zero-on-fix
git add pyproject.toml .github/workflows/build.yml codecov.yml
git commit --no-gpg-sign -m "ci: combined-backend coverage job + Codecov (gate not yet enforcing)"
```

---

### Task 2: Core package to 100%

**Files:**
- Create: `tests/core/test_numerics.py`
- Modify: `tests/core/test_config.py`
- Modify: `tests/core/test_reference_activations.py`
- Modify: `tests/core/test_reference_dense.py`
- Modify: `tests/core/test_init.py`

**Interfaces:**
- Consumes: `mononet.core.numerics.default_atol(dtype)->float`, `default_rtol(dtype)->float`; `mononet.core.config.{MonoConfig,MonoResidualConfig}`; `mononet.core.reference.{base_activation,apply_gate,monotonic_dense}`; `mononet.core.init._bisect`.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Confirm the core gap is real**

```bash
uv run coverage erase
MONONET_TEST_BACKEND=torch uv run pytest tests/core -q
uv run coverage report --include="mononet/core/*" --show-missing
```
Expected: `numerics.py` 0% (missing 4–29), `config.py` missing 37/98/100, `reference.py` missing 43-45/75/117, `init.py` missing 87.

- [ ] **Step 2: Write `tests/core/test_numerics.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for mononet.core.numerics tolerance helpers."""

from __future__ import annotations

import numpy as np

from mononet.core.numerics import (
    ATOL_FLOAT32,
    ATOL_FLOAT64,
    RTOL_FLOAT32,
    RTOL_FLOAT64,
    default_atol,
    default_rtol,
)


def test_default_tolerances_for_float64() -> None:
    assert default_atol(np.float64) == ATOL_FLOAT64
    assert default_rtol(np.float64) == RTOL_FLOAT64


def test_default_tolerances_for_float32() -> None:
    assert default_atol(np.float32) == ATOL_FLOAT32
    assert default_rtol(np.float32) == RTOL_FLOAT32


def test_non_float64_dtype_falls_back_to_float32_tolerances() -> None:
    # any dtype that is not float64 uses the float32 tolerances
    assert default_atol(np.float16) == ATOL_FLOAT32
    assert default_rtol(np.float16) == RTOL_FLOAT32
```

- [ ] **Step 3: Add config error-path tests to `tests/core/test_config.py`**

Append:

```python
def test_mono_config_rejects_bad_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        MonoConfig(units=4, mode="bogus")  # type: ignore[arg-type]


def test_mono_residual_config_rejects_bad_units_and_mode() -> None:
    with pytest.raises(ValueError, match="units must be positive"):
        MonoResidualConfig(units=0, activation=ActivationSpec("relu"))
    with pytest.raises(ValueError, match="mode must be"):
        MonoResidualConfig(
            units=4, mode="bogus", activation=ActivationSpec("relu")
        )  # type: ignore[arg-type]
```

- [ ] **Step 4: Add reference activation/gate error-path tests to `tests/core/test_reference_activations.py`**

Append:

```python
def test_identity_activation_is_passthrough() -> None:
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(ref.base_activation("identity", x), x)


def test_unknown_activation_raises() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        ref.base_activation("bogus", np.zeros(3))  # type: ignore[arg-type]


def test_unknown_gate_token_raises() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        ref.apply_gate("bogus", np.zeros(3))
```

- [ ] **Step 5: Add the reference bad-mode test to `tests/core/test_reference_dense.py`**

First check the existing imports at the top of `tests/core/test_reference_dense.py`; it already imports numpy and the reference module. Append (adjust the `ref`/import alias to match the file — it uses `from mononet.core import reference as ref` or similar; if it imports `monotonic_dense` directly, call that instead):

```python
def test_monotonic_dense_rejects_unknown_mode() -> None:
    import numpy as np
    import pytest

    from mononet.core import reference as ref
    from mononet.core.types import ActivationSpec

    x = np.zeros((2, 3))
    w = np.ones((3, 4))
    b = np.zeros(4)
    with pytest.raises(ValueError, match="mode must be"):
        ref.monotonic_dense(x, w, b, "bogus", ActivationSpec("relu"))
```

- [ ] **Step 6: Add the `_bisect` non-convergence test to `tests/core/test_init.py`**

Append (covers `init.py:87`, the fall-through when the loop exhausts `iters` without reaching `tol`):

```python
def test_bisect_returns_midpoint_when_iterations_exhausted() -> None:
    from mononet.core.init import _bisect

    # Root of (x - 1/3) is 1/3; with tol=0 the |fmid| < tol test never fires,
    # so after `iters` steps the loop falls through to `return 0.5*(lo+hi)`.
    root = _bisect(lambda x: x - 1.0 / 3.0, 0.0, 1.0, tol=0.0, iters=5)
    assert abs(root - 1.0 / 3.0) < 0.1
```

- [ ] **Step 7: Run the core suite and confirm 100% on `mononet/core/*`**

```bash
uv run coverage erase
MONONET_TEST_BACKEND=torch uv run pytest tests/core -q
uv run coverage report --include="mononet/core/*" --show-missing
```
Expected: every `mononet/core/*` row shows **100%**, no missing lines.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format tests/core
uv run ruff check --exit-non-zero-on-fix
git add tests/core
git commit --no-gpg-sign -m "test(core): cover numerics, config/reference error paths, bisect fallback"
```

---

### Task 3: Torch backend to 100%

**Files:**
- Create: `tests/torch/test_coverage_gaps.py`

**Interfaces:**
- Consumes: `mononet.torch._kernels.{activation,gate,monotonic_dense}`; `mononet.torch.{MonoLinear,MonoResidual,MonoInput}`; `mononet.core.types.InitSpec`.

- [ ] **Step 1: Confirm the torch gap**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=torch uv run pytest tests/core tests/torch tests/equivalence -q
uv run coverage report --include="mononet/torch/*" --show-missing
```
Expected: `_kernels.py` missing 40/69/104, `layers.py` missing 50/52/185/224.

- [ ] **Step 2: Write `tests/torch/test_coverage_gaps.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Cover torch kernel error paths and layer branches not hit elsewhere."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mononet.core.types import InitSpec  # noqa: E402
from mononet.torch import MonoInput, MonoLinear, MonoResidual  # noqa: E402
from mononet.torch import _kernels  # noqa: E402


def test_kernel_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _kernels.activation("bogus", torch.zeros(3))


def test_kernel_gate_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        _kernels.gate("bogus", torch.zeros(()))


def test_kernel_dense_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _kernels.monotonic_dense(
            torch.zeros(2, 3), torch.ones(3, 4), torch.zeros(4), "bogus", "relu"
        )


def test_init_weight_accepts_initspec_with_seed() -> None:
    # InitSpec instance (not None/str) hits the `spec = init` branch;
    # a non-None seed hits the `torch.manual_seed(spec.seed)` branch.
    layer = MonoLinear(3, 5, activation="relu", init=InitSpec(scheme="he_normal", seed=0))
    assert layer(torch.zeros(2, 3)).shape == (2, 5)


def test_residual_accepts_callable_f_factory() -> None:
    # A plain callable (not an nn.Module) hits the `self.F = F(units)` branch.
    block = MonoResidual(4, 4, F=lambda u: MonoLinear(u, u, activation="relu"))
    assert block(torch.zeros(2, 4)).shape == (2, 4)


def test_mono_input_accepts_scalar_direction() -> None:
    # int direction (not a MonotonicityMask) hits the scalar branch.
    layer = MonoInput(-1)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(layer(x), -x)
```

- [ ] **Step 3: Run torch coverage and confirm 100% on `mononet/torch/*`**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=torch uv run pytest tests/core tests/torch tests/equivalence -q
uv run coverage report --include="mononet/torch/*" --show-missing
```
Expected: `mononet/torch/_kernels.py` and `mononet/torch/layers.py` both **100%**.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff format tests/torch/test_coverage_gaps.py
uv run ruff check --exit-non-zero-on-fix
git add tests/torch/test_coverage_gaps.py
git commit --no-gpg-sign -m "test(torch): cover kernel error paths and layer branches"
```

---

### Task 4: JAX backend to 100%

**Files:**
- Create: `tests/jax/test_coverage_gaps.py`

**Interfaces:**
- Consumes: `mononet.jax._kernels.{activation,gate,monotonic_dense}`; `mononet.jax.{MonoLinear,MonoResidual,MonoInput}`; `flax.nnx.Rngs`.

- [ ] **Step 1: Confirm the jax gap**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=jax uv run pytest tests/jax tests/equivalence -q
uv run coverage report --include="mononet/jax/*" --show-missing
```
Expected: `_kernels.py` missing 30/56/91, `layers.py` missing 214/253.

- [ ] **Step 2: Write `tests/jax/test_coverage_gaps.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Cover jax kernel error paths and layer branches not hit elsewhere."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
nnx = pytest.importorskip("flax.nnx")
import jax.numpy as jnp  # noqa: E402

from mononet.jax import MonoInput, MonoLinear, MonoResidual  # noqa: E402
from mononet.jax import _kernels  # noqa: E402


def test_kernel_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _kernels.activation("bogus", jnp.zeros(3))


def test_kernel_gate_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        _kernels.gate("bogus", jnp.zeros(()))


def test_kernel_dense_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _kernels.monotonic_dense(
            jnp.zeros((2, 3)), jnp.ones((3, 4)), jnp.zeros(4), "bogus", "relu"
        )


def test_residual_accepts_callable_f_factory() -> None:
    rngs = nnx.Rngs(0)
    # A plain callable (not an nnx.Module) hits the `self.F = F(units)` branch.
    block = MonoResidual(
        4, 4, F=lambda u: MonoLinear(u, u, activation="relu", rngs=rngs), rngs=rngs
    )
    assert block(jnp.zeros((2, 4))).shape == (2, 4)


def test_mono_input_accepts_scalar_direction() -> None:
    layer = MonoInput(-1)
    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(layer(x), -x)
```

- [ ] **Step 3: Run jax coverage and confirm 100% on `mononet/jax/*`**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=jax uv run pytest tests/jax tests/equivalence -q
uv run coverage report --include="mononet/jax/*" --show-missing
```
Expected: `mononet/jax/_kernels.py` and `mononet/jax/layers.py` both **100%**.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff format tests/jax/test_coverage_gaps.py
uv run ruff check --exit-non-zero-on-fix
git add tests/jax/test_coverage_gaps.py
git commit --no-gpg-sign -m "test(jax): cover kernel error paths and layer branches"
```

---

### Task 5: Keras backend to 100%

**Files:**
- Create: `tests/keras/test_coverage_gaps.py`

**Interfaces:**
- Consumes: `mononet.keras._kernels.{activation,gate,monotonic_dense}`; `mononet.keras.{MonoDense,MonoResidual,MonoInput}`; `keras.ops`. Note: keras `MonoResidual` takes `units` as its first positional arg (no `in_features`), unlike torch/jax.

- [ ] **Step 1: Confirm the keras gap**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=keras uv run pytest tests/keras tests/equivalence -q
uv run coverage report --include="mononet/keras/*" --show-missing
```
Expected: `_kernels.py` missing 30/56/93, `layers.py` missing 267-278/301.

- [ ] **Step 2: Write `tests/keras/test_coverage_gaps.py`**

```python
# SPDX-License-Identifier: Apache-2.0
"""Cover keras kernel error paths and layer branches not hit elsewhere."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

from mononet.keras import MonoInput, MonoResidual  # noqa: E402
from mononet.keras import _kernels  # noqa: E402


def test_kernel_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _kernels.activation("bogus", ops.zeros(3))


def test_kernel_gate_rejects_unknown_token() -> None:
    with pytest.raises(ValueError, match="unknown gate token"):
        _kernels.gate("bogus", ops.zeros(()))


def test_kernel_dense_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be"):
        _kernels.monotonic_dense(
            ops.zeros((2, 3)), ops.ones((3, 4)), ops.zeros(4), "bogus", "relu"
        )


def test_residual_get_config_roundtrips() -> None:
    block = MonoResidual(4, mode="switch", activation="relu")
    block(ops.zeros((2, 4)))  # build so config fields are populated
    cfg = block.get_config()
    assert cfg["units"] == 4
    assert cfg["mode"] == "switch"
    assert cfg["activation"] == "relu"
    assert cfg["alpha_gate"] == "shifted_elu"
    assert cfg["beta_gate"] == "scaled_elu"


def test_mono_input_accepts_scalar_direction() -> None:
    layer = MonoInput(-1)
    x = ops.convert_to_tensor(np.array([[1.0, 2.0, 3.0]]))
    assert bool(ops.all(layer(x) == -x))
```

- [ ] **Step 3: Run keras coverage and confirm 100% on `mononet/keras/*`**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=keras uv run pytest tests/keras tests/equivalence -q
uv run coverage report --include="mononet/keras/*" --show-missing
```
Expected: `mononet/keras/_kernels.py` and `mononet/keras/layers.py` both **100%**.

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff format tests/keras/test_coverage_gaps.py
uv run ruff check --exit-non-zero-on-fix
git add tests/keras/test_coverage_gaps.py
git commit --no-gpg-sign -m "test(keras): cover kernel error paths, get_config, layer branches"
```

---

### Task 6: Turn the gate on

**Files:**
- Modify: `.github/workflows/build.yml` (add `--fail-under=100`; add `coverage` to `check.needs`)

**Interfaces:**
- Consumes: the `coverage` job from Task 1 and 100% coverage established by Tasks 2–5.

- [ ] **Step 1: Verify the full combined run is at 100% locally FIRST**

```bash
uv run coverage erase
KERAS_BACKEND=jax MONONET_TEST_BACKEND=torch uv run pytest tests/core tests/torch tests/equivalence tests/examples tests/test_top_level_imports.py -q
KERAS_BACKEND=jax MONONET_TEST_BACKEND=jax   uv run pytest tests/jax   tests/equivalence -q
KERAS_BACKEND=jax MONONET_TEST_BACKEND=keras uv run pytest tests/keras tests/equivalence -q
uv run coverage report --show-missing --fail-under=100
```
Expected: TOTAL **100%**, command exits **0**. If any line is still missing, STOP and add the covering test before proceeding.

- [ ] **Step 2: Add `--fail-under=100` to the `coverage` job**

In `.github/workflows/build.yml`, change the `coverage report` line in the `coverage` job's run block to:

```yaml
          coverage report --show-missing --fail-under=100
```

- [ ] **Step 3: Add `coverage` to `check.needs`**

In `.github/workflows/build.yml`, the `check` job's `needs` (currently line 104) becomes:

```yaml
    needs: [static-analysis, pre-commit, docs-smoke, test, coverage]
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/build.yml
git commit --no-gpg-sign -m "ci: enforce 100% coverage (--fail-under=100, gate via check)"
```

- [ ] **Step 5: Push the branch and open the PR**

```bash
git push -u origin spec/coverage-100-enforcement
```
Then open a PR per `PULL_REQUEST_GUIDE.md`. On the PR, confirm: the `coverage` job passes, Codecov reports **100%** project and **100%** patch.

- [ ] **Step 6: Post-merge action item (repo settings, NOT code)**

To make Codecov statuses *required* for merge, a maintainer adds `codecov/project` and `codecov/patch` to branch-protection required checks (the hard gate already blocks via `check`). Flag this in the PR description. This can be done with `gh api` if the maintainer wants it automated.

---

## Notes for the implementer

- **Do not add `--cov` flags to the per-task `pytest` commands** — `addopts` already injects `--cov=mononet --cov=docs/examples --cov-append --cov-branch`. Always `coverage erase` before a measurement run so `--cov-append` starts clean.
- **`uv sync` is exact and will prune backend extras.** Never run `uv sync --group X` without also passing `--extra all-cpu`; if the environment loses `torch`/`jax`/`keras`, restore with `uv sync --extra all-cpu --group dev --group bench`.
- If a `tests/core/test_reference_dense.py` import alias differs from the snippet in Task 2 Step 5, match the file's existing style; the assertion is what matters.
