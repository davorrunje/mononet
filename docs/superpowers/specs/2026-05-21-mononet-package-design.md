# mononet — Multi-Framework Package Design

**Date:** 2026-05-21
**Author:** Davor Runje
**Status:** Approved (brainstorming output); pending implementation plan.

> **Note (2026-06-27):** Sub-project A was redesigned after initial scaffolding. The current
> implementation is described in
> [2026-06-27-A-core-algorithm-and-backends-design.md](2026-06-27-A-core-algorithm-and-backends-design.md).
> Sections below reflect the original design; where they conflict with the redesign spec, the
> redesign spec takes precedence.

## 1. Goals & non-goals

### Goals
- Publish `mononet` to public PyPI as the canonical implementation of the
  unconstrained monotonic neural network technique from
  Runje & Shankaranarayana (2023), *Constrained Monotonic Neural Networks*,
  ICML 2023 — <https://arxiv.org/abs/2205.11775>.
- First-class, idiomatic support for **PyTorch**, **JAX (Flax NNX)**, and
  **Keras 3** under a single installable package with optional extras.
- Reproduce the paper's benchmarks as MkDocs-hosted notebooks; compare against
  `airtai/monotonic-nn` (the paper's original implementation) as the numerical
  baseline. That package is installed manually at benchmark-execution time via
  `--no-deps` (its old `typing-extensions` pin conflicts with modern deps); it
  is not listed in the `bench` dependency group.
- Repurpose the existing cookiecutter scaffold: keep its strict
  static-analysis + docs + release pipeline; strip company-specific bits
  (1Password CLI integration, the synthpop private PyPI index, Linear
  workflow, Codecov, the second "partner" devcontainer flavor).

### Non-goals (v0.x)
- No GPU CI (free runners do not provide GPUs; local devcontainers cover
  GPU work).
- No TensorFlow-native backend (Keras 3 with TF backend covers this).
- No ONNX or TFLite export pipelines.
- No distributed-training utilities — base layers/models only.
- No training-loop library; users plug layers into their own
  (PyTorch Lightning, Flax, Keras Trainer, etc.).

## 2. Package layout

```
mononet/                                # repo root
├── mononet/                            # the installable package
│   ├── __init__.py                     # public API surface (no eager backend imports);
│   │                                   # exposes __version__ via importlib.metadata
│   ├── core/                           # framework-agnostic
│   │   ├── __init__.py
│   │   ├── reference.py                # NumPy reference impl of the monotonic primitive
│   │   ├── config.py                   # dataclass configs shared by backends
│   │   ├── types.py                    # MonotonicityMask, ActivationSpec, etc.
│   │   └── numerics.py                 # tolerances, dtype helpers, RNG seeding
│   ├── torch/
│   │   ├── __init__.py
│   │   ├── _kernels.py                 # private: framework-native math
│   │   └── layers.py                   # public: MonoLinear, MonoResidual, MonoInput
│   ├── jax/
│   │   ├── __init__.py                 # Flax NNX
│   │   ├── _kernels.py
│   │   └── layers.py                   # public: MonoLinear, MonoResidual, MonoInput
│   ├── keras/
│   │   ├── __init__.py                 # Keras 3, backend-agnostic via keras.ops
│   │   ├── _kernels.py
│   │   └── layers.py                   # public: MonoDense, MonoResidual, MonoInput
│   └── py.typed                        # PEP 561 marker
├── tests/
│   ├── core/                           # NumPy reference + property tests
│   ├── torch/                          # torch-only tests
│   ├── jax/
│   ├── keras/
│   └── equivalence/                    # cross-backend numerical equivalence tests
│       └── cases/                      # committed JSON test vectors
├── docs/                               # MkDocs site
│   ├── docs/
│   │   ├── index.md
│   │   ├── api/                        # mkdocstrings-generated
│   │   ├── guides/
│   │   │   ├── pytorch.md
│   │   │   ├── jax.md
│   │   │   └── keras.md
│   │   ├── concepts/
│   │   │   ├── monotonicity.md
│   │   │   └── layers.md
│   │   ├── benchmarks/                 # reproducing-the-paper notebooks
│   │   │   ├── index.md
│   │   │   └── *.ipynb                 # rendered by mkdocs-jupyter (execute: false)
│   │   └── about/
│   │       ├── license.md
│   │       ├── changelog.md
│   │       └── citation.md
│   └── mkdocs.yml
├── .devcontainer/
│   ├── shared/                         # install scripts shared across flavors
│   ├── default/                        # CPU (Python 3.13), all backends installable
│   ├── gpu-torch/                      # nvidia/cuda + torch CUDA wheels
│   ├── gpu-jax/                        # nvidia/cuda + jax[cuda12]
│   └── gpu-keras/                      # nvidia/cuda + keras + jax[cuda12]
├── .github/
│   ├── workflows/
│   │   ├── build.yml                   # lint + static + per-backend test jobs
│   │   ├── docs.yml                    # mkdocs gh-deploy via mike on tags
│   │   ├── publish.yml                 # public PyPI via trusted publishing (OIDC)
│   │   ├── bump-version.yml
│   │   └── codeql.yml
│   ├── ISSUE_TEMPLATE/
│   ├── dependabot.yml
│   └── pull_request_template.md
├── tools/                              # dev scripts
├── LICENSE                             # PolyForm Noncommercial 1.0.0
├── NOTICE.md                           # patent reservation + commercial-license contact
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CLAUDE.md
├── PULL_REQUEST_GUIDE.md
└── pyproject.toml
```

### Public import surface

Lazy — each user pays only the backend they ask for. `import mononet` works
even with zero backends installed:

```python
from mononet.torch import MonoLinear, MonoResidual, MonoInput   # imports torch only
from mononet.jax   import MonoLinear, MonoResidual, MonoInput   # imports jax only
from mononet.keras import MonoDense, MonoResidual, MonoInput    # imports keras only
from mononet.core.reference import monotonic_dense              # NumPy, no framework needed
```

`mononet/__init__.py` exports `__version__` and `mononet.core` symbols only —
no eager backend imports.

### Naming convention

Each backend mirrors its host framework's vocabulary for the analogous
unconstrained layer.

| Concept                       | Core (NumPy)                      | PyTorch                        | JAX (Flax NNX)                 | Keras 3                        |
|-------------------------------|-----------------------------------|--------------------------------|--------------------------------|--------------------------------|
| Single monotonic layer        | `monotonic_dense` (function)      | `MonoLinear(nn.Module)`        | `MonoLinear(nnx.Module)`       | `MonoDense(keras.Layer)`       |
| Residual monotonic block      | `monotonic_residual` (function)   | `MonoResidual(nn.Module)`      | `MonoResidual(nnx.Module)`     | `MonoResidual(keras.Layer)`    |
| Input projection / gating     | —                                 | `MonoInput(nn.Module)`         | `MonoInput(nnx.Module)`        | `MonoInput(keras.Layer)`       |
| Monotonicity spec             | `MonotonicityMask` — `{-1,+1}` per feature (shared in `mononet.core.types`); enforced by `MonoInput` |

Rules:
- PyTorch and JAX/Flax: `MonoLinear` (both frameworks call the standard
  analog `Linear`).
- Keras: `MonoDense` (Keras calls it `Dense`).
- `MonoResidual` and `MonoInput` share one name across all three backends.
- There are no composed model classes (`MonoMLP`/`MonoFeatureBlock` were
  dropped); users stack layers using the framework's native `Sequential`.
- Pure-function NumPy reference uses `snake_case` (`monotonic_dense`,
  `monotonic_residual`) to signal stateless reference implementations, not
  layers.
- Shared config/typing classes (`MonotonicityMask`, `ActivationSpec`,
  `InitSpec`, `MonoConfig`, `MonoResidualConfig`) live in `mononet.core`
  and are imported by all backends.

### Files removed from the cookiecutter

- `.linear.toml`
- `LINEAR_GUIDE.md`
- `.claude/skills/linear-cli/`
- `codecov.yml`
- `.devcontainer/partner/` (the second cookiecutter flavor)
- `.devcontainer/default/initialize_devcontainer.sh` (1Password)
- `.devcontainer/default/devcontainer.env`, `devcontainer.env.tmp`
- `.devcontainer/default/post-start.sh` (only existed to clean up `.env.tmp`)
- `[[tool.uv.index]] synthpop-pkgs` and `override-dependencies` in
  `pyproject.toml`
- All `UV_INDEX_SYNTHPOP_PKGS_*` env blocks in workflows
- `mononet/__init__.py`'s `HelloWorld` placeholder

## 3. Backend architecture & equivalence testing

### Per-backend implementation pattern

Each backend module follows the same internal shape so contributors can move
between them:

```
mononet/<backend>/
├── __init__.py        # public exports: MonoLinear/MonoDense, MonoResidual, MonoInput
├── _kernels.py        # private — the math, in the framework's native ops
└── layers.py          # public — MonoLinear/MonoDense, MonoResidual, MonoInput
```

`_kernels.py` contains pure functional code (e.g. for PyTorch: a function
taking tensors, returning a tensor — no `nn.Module` state). `layers.py`
wraps the kernel in the framework-idiomatic stateful container. This
separation matters because:
- the `_kernels` layer is what gets validated against the NumPy reference,
- the wrapper layer is what gets validated against framework idioms
  (serialization, `state_dict`, `nnx.split`, `keras.saving`).

### Configuration via stdlib dataclasses

Configs are simple value objects: a few typed fields, modest validation,
JSON round-trip for reproducibility. `dataclasses` from the standard library
covers this without adding a runtime dependency. Pydantic was considered
and rejected because configs are not complex enough to justify pulling in
`pydantic-core` (a Rust binary) and risking version conflicts with other
ML libraries.

```python
# mononet/core/config.py
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class MonoConfig:
    in_features: int
    out_features: int
    monotonicity: MonotonicityMask       # +1 / -1 per input feature (no 0)
    activation: ActivationSpec           # "relu" | "elu" | "selu" | "softplus"
    init: InitSpec                       # seed, scheme (default: he_normal)

    def __post_init__(self) -> None:
        # one-time validation: in_features > 0, mask values in {-1,+1}, etc.
        ...

@dataclass(frozen=True, slots=True)
class MonoResidualConfig:
    ...  # config for MonoResidual blocks
```

Each backend layer's `__init__` accepts either the loose kwargs or a
`MonoConfig` / `MonoResidualConfig`. Configs serialize to JSON via `dataclasses.asdict()`
for benchmark reproducibility; load via a `from_dict` constructor (~15
lines total in `mononet.core.config`).

### NumPy reference (`mononet.core.reference`)

Pure functions, no framework imports:

```python
def monotonic_dense(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> np.ndarray: ...
```

This is the arithmetic ground truth. The paper's equations live here, with
citations to the arxiv paper. Every backend kernel is asserted equivalent
to this within `atol=1e-5, rtol=1e-5` (float32) or `1e-6` (float64) across
a fixed property-test grid.

### Equivalence testing strategy (`tests/equivalence/`)

For each layer type, a battery of `(shape, dtype, mask, activation, seed)`
cases is pre-generated and committed as JSON in `tests/equivalence/cases/`
so the same vectors are checked every CI run — no flaky random seeds.

For each case:

1. Compute output with the NumPy reference.
2. Compute output with each installed backend
   (skipped if backend not installed via `pytest.importorskip`).
3. Assert all backends agree with the reference and with each other.
4. Compute gradients: numerical (NumPy finite differences) vs each
   backend's autograd. Assert agreement.

```python
@pytest.mark.parametrize("case", load_cases("mono_linear"))
def test_mono_linear_matches_reference(case, backend):
    expected = reference.monotonic_dense(**case.inputs)
    got = backend.run_mono_linear(**case.inputs)
    np.testing.assert_allclose(got, expected, atol=case.atol, rtol=case.rtol)
```

`backend` is a parametrized fixture iterating over `{torch, jax, keras}`,
skipping any not installed.

### Why this works

- **Idiomatic backends** — wrapper layers are pure framework code, look
  native to a PyTorch/JAX/Keras user.
- **One source of truth** — paper math is in `mononet.core.reference`, not
  duplicated across backends.
- **Cheap to add a 4th backend** — implement `_kernels.py` against the
  reference, the equivalence harness validates it automatically.
- **CI catches drift early** — if anyone tweaks a backend's `_kernels.py`,
  the equivalence test fails before merge.

### JAX-specific choice: Flax NNX

Defaulting to **Flax NNX** (the new Flax API, released 2024) over Equinox or
raw stax for the JAX backend:
- Official Google/Flax-team recommended path going forward.
- Plays well with `pytree`s and shares mental model with PyTorch (objects
  with attributes) — easier for paper readers to follow.
- `nnx.Module` integrates with `jax.jit` / `jax.grad` cleanly.

### Keras 3 backend choice

`mononet.keras` uses **`keras.ops`** (the backend-agnostic op layer in
Keras 3). `MonoDense` works whether the user runs Keras with TF, JAX, or
PyTorch backend without us writing three versions. Tests run with
`KERAS_BACKEND=jax` by default (fastest CI install). A smoke test verifies
the other two.

## 4. Devcontainers (4 flavors)

### Layout

```
.devcontainer/
├── shared/
│   ├── install_common_tools.sh         # uv, git-lfs, gh, common apt deps
│   ├── install_uv.sh                   # uv on CUDA base (no system Python)
│   ├── setup_path.sh
│   └── post-create.sh                  # uv sync + pre-commit install
├── default/                            # CPU, all backends
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   └── setup.sh
├── gpu-torch/
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   └── setup.sh
├── gpu-jax/
│   ├── Dockerfile
│   ├── devcontainer.json
│   ├── docker-compose.yml
│   └── setup.sh
└── gpu-keras/
    ├── Dockerfile
    ├── devcontainer.json
    ├── docker-compose.yml
    └── setup.sh
```

No more `initialize_devcontainer.sh` (1Password), no `devcontainer.env.tmp`,
no `op signin`. Secrets that researchers might want (e.g. HuggingFace token)
are documented in `CONTRIBUTING.md` as environment variables they set in
their shell, not pulled from a vault.

### `default` (CPU)

- **Base**: `mcr.microsoft.com/devcontainers/python:3.13`.
- **Installs**: `pip install -e ".[all,dev,docs,lint]"` — all three backends
  CPU-only.
- Used for: writing code, running unit tests + equivalence tests locally on
  CPU, building docs.

### `gpu-torch`

- **Base**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
- **Python**: installed via `uv python install 3.13`. The NVIDIA image has
  no Python.
- **Installs**: `pip install -e ".[torch-gpu,dev,docs,lint]"`.
- **runArgs**: `--gpus=all` so the container sees the host's GPU.
- Used for: GPU benchmarks against the paper's PyTorch baseline.

### `gpu-jax`

- **Base**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
- **Installs**: `pip install -e ".[jax-gpu,dev,docs,lint]"` (`jax[cuda12]`).
- **runArgs**: `--gpus=all`.

### `gpu-keras`

- **Base**: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
- **Installs**: `pip install -e ".[keras-gpu,dev,docs,lint]"`. Resolves to
  `keras + jax[cuda12]` (Keras with the JAX GPU backend by default —
  fastest install, cleanest interaction with `keras.ops`).
- Env var: `KERAS_BACKEND=jax`.
- `setup.sh` documents how to re-install with `KERAS_BACKEND=torch` against
  PyTorch CUDA wheels if preferred.

### `shared/` script reuse

All four `setup.sh` scripts delegate to `shared/post-create.sh`, which is
the only place that runs `uv sync`, installs pre-commit hooks, and
configures git LFS. Per-flavor `setup.sh` only handles the framework-specific
install line. CUDA toolkit upgrades or uv version bumps are a one-file
change.

### Dockerfile per GPU flavor

Each `gpu-*` flavor gets a small `Dockerfile` (~30 lines) that:
1. Starts from `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.
2. Installs `curl ca-certificates git git-lfs gnupg`.
3. Installs `uv` via the official install script.
4. Installs Python via `uv python install 3.13`.

The three Dockerfiles share ~95% of their content. Kept duplicated rather
than templated; the per-framework `pip install` line lives in `setup.sh`,
not the Dockerfile, so the Dockerfiles themselves are nearly identical
and easy to keep in sync manually.

### VS Code "Reopen in Container" picker

Lists 4 entries (controlled by `devcontainer.json:name`):
- `python-3.13`
- `python-3.13 — GPU (PyTorch)`
- `python-3.13 — GPU (JAX)`
- `python-3.13 — GPU (Keras)`

## 5. `pyproject.toml` structure

### Build & metadata

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "mononet"
version = "0.0.0"                                  # bumped via bump-version.yml
description = "Unconstrained monotonic neural networks for PyTorch, JAX, and Keras"
authors = [{ name = "Davor Runje", email = "davor.runje@fer.hr" }]
maintainers = [{ name = "Davor Runje", email = "davor.runje@fer.hr" }]
requires-python = ">=3.11,<3.14"
readme = "README.md"
license = "LicenseRef-PolyForm-Noncommercial-1.0.0"
license-files = ["LICENSE", "NOTICE.md"]
keywords = ["monotonic", "neural-network", "pytorch", "jax", "keras", "deep-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
    # No OSI license classifier — PolyForm is not OSI-approved.
]
dependencies = [
    "numpy>=1.26",
    "typing-extensions>=4.12; python_version<'3.12'",
]

[project.urls]
Homepage      = "https://github.com/davorrunje/mononet"
Documentation = "https://davorrunje.github.io/mononet"
Repository    = "https://github.com/davorrunje/mononet"
Issues        = "https://github.com/davorrunje/mononet/issues"
Changelog     = "https://github.com/davorrunje/mononet/blob/main/CHANGELOG.md"
Paper         = "https://arxiv.org/abs/2205.11775"
Patent        = "https://patents.justia.com/patent/11551063"
```

### Extras (multi-backend story)

```toml
[project.optional-dependencies]
# CPU backends
torch       = ["torch>=2.4"]
jax         = ["jax>=0.4.30", "flax>=0.10"]               # Flax NNX
keras       = ["keras>=3.5", "jax>=0.4.30"]               # keras+jax CPU by default
all         = ["mononet[torch,jax,keras]"]

# GPU backends — used by the gpu-* devcontainers and power-users.
# Actual CUDA wheels resolve via the framework's own index URLs; these
# extras pin the package, the user/devcontainer configures the index.
torch-gpu   = ["torch>=2.4"]
jax-gpu     = ["jax[cuda12]>=0.4.30", "flax>=0.10"]
keras-gpu   = ["keras>=3.5", "jax[cuda12]>=0.4.30"]
```

### Dependency groups (dev-time only, never on PyPI)

```toml
[dependency-groups]
dev = [
    "ipython",
    "ipykernel",
    "mypy==1.20.1",
    "pytest==9.0.3",
    "pytest-asyncio==1.3.0",
    "pytest-cov==7.1.0",
    "hypothesis>=6.115",                       # property-based equivalence tests
]
docs = [
    "mkdocs-material==9.7.6",
    "mkdocstrings[python]==1.0.3",
    "mkdocs-literate-nav==0.6.3",
    "mkdocs-glightbox==0.5.2",
    "mkdocs-jupyter>=0.25",                    # render benchmark notebooks
    "mike==2.1.4",                             # versioned docs
    "mkdocs-git-revision-date-localized-plugin==1.5.1",
    "mkdocs-minify-plugin==0.8.0",
]
lint = [
    "pre-commit==4.5.1",
    "ruff==0.15.10",
    "bandit==1.9.4",
    "semgrep==1.159.0",
    "codespell==2.4.2",
    "detect-secrets==1.5.0",
]
bench = [                                      # for running benchmark notebooks
    "scikit-learn>=1.5",
    "pandas>=2.2",
    "matplotlib>=3.9",
    # airtai/monotonic-nn is NOT listed here: its old typing-extensions pin
    # conflicts with modern numpy/torch/jax/keras. Install it manually at
    # benchmark-execution time with --no-deps (see tools/execute-benchmarks.sh).
]
```

`devdocs` renamed to `docs`. `bench` is new — installed locally by the
maintainer when re-executing benchmark notebooks.

### `[tool.uv]`

```toml
[tool.uv]
default-groups = ["dev", "docs", "lint"]
```

Removed: `[[tool.uv.index]] synthpop-pkgs` and `override-dependencies`.

### Tool configuration

- `[tool.mypy]`: `python_version = "3.11"`. Remove `plugins = ["pydantic.mypy"]`.
- Remove `[tool.pydantic-mypy]` block entirely.
- `[tool.ruff]`: `target-version = "py311"`.
- `[tool.pytest.ini_options]` `addopts`: drop `--cov-report=xml` (no Codecov);
  keep `--cov=mononet --cov-append --cov-branch --cov-report=term-missing`.
- Everything else (`bandit`, ruff lint selections, per-file ignores) kept
  verbatim.

### Hatch wheel includes

```toml
[tool.hatch.build.targets.sdist]
include = ["mononet", "LICENSE", "NOTICE.md", "README.md", "CHANGELOG.md"]

[tool.hatch.build.targets.wheel]
include = ["mononet"]
```

## 6. CI workflows

### `.github/workflows/build.yml` — lint, static, per-backend tests

```yaml
name: Build
on:
  push: { branches: [main] }
  pull_request:
  merge_group:

permissions:
  contents: read

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}-${{ github.head_ref }}
  cancel-in-progress: true

jobs:

  static-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with: { python-version: "3.13" }
      - uses: astral-sh/setup-uv@v7
      - run: uv pip install --system -e ".[all]" --group=dev --group=lint
      - run: ./tools/static-analysis.sh

  pre-commit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with: { python-version: "3.13" }
      - uses: astral-sh/setup-uv@v7
      - run: uv pip install --system -e ".[all]" --group=dev --group=lint
      - uses: actions/cache@v5
        with:
          path: ~/.cache/pre-commit
          key: pre-commit|${{ hashFiles('.pre-commit-config.yaml') }}
      - uses: pre-commit/action@v3.0.1
        with: { extra_args: --hook-stage manual --all-files }

  docs-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with: { python-version: "3.13" }
      - uses: astral-sh/setup-uv@v7
      - run: uv pip install --system -e ".[all]" --group=docs
      - run: cd docs && mkdocs build --strict

  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest]
        python-version: ["3.11", "3.12", "3.13"]
        backend: [torch, jax, keras]
        include:
          - { os: macos-latest,   python-version: "3.13", backend: torch }
          - { os: macos-latest,   python-version: "3.13", backend: jax }
          - { os: macos-latest,   python-version: "3.13", backend: keras }
          - { os: windows-latest, python-version: "3.13", backend: torch }
          - { os: windows-latest, python-version: "3.13", backend: jax }
          - { os: windows-latest, python-version: "3.13", backend: keras }

    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with: { python-version: ${{ matrix.python-version }} }
      - uses: astral-sh/setup-uv@v7
      - run: uv pip install --system -e ".[${{ matrix.backend }}]" --group=dev
      - run: pytest tests/core tests/${{ matrix.backend }} tests/equivalence -v
        env:
          MONONET_TEST_BACKEND: ${{ matrix.backend }}

  check:                                          # required-status branch protection target
    if: always()
    runs-on: ubuntu-latest
    needs: [static-analysis, pre-commit, docs-smoke, test]
    steps:
      - uses: re-actors/alls-green@release/v1
        with: { jobs: ${{ toJSON(needs) }} }
```

**Job count**: 3×3 (Ubuntu) + 3 (macOS) + 3 (Windows) + 3 (static / pre-commit / docs-smoke) = **18 jobs per PR**.

`MONONET_TEST_BACKEND` tells the equivalence harness which backend to
exercise; the suite uses `pytest.importorskip` to skip any other backend
not installed in the current job. Each backend is independently validated
against the same NumPy reference, so passing the per-backend equivalence
tests transitively implies inter-backend agreement — no separate nightly
all-backend job is needed.

### `.github/workflows/docs.yml` — docs build & deploy

```yaml
name: Docs
on:
  push:
    branches: [main]
    tags: ['v*.*.*']
permissions:
  contents: write                                 # mike pushes to gh-pages
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
        with: { fetch-depth: 0 }                  # mike needs history
      - uses: actions/setup-python@v6
        with: { python-version: "3.13" }
      - uses: astral-sh/setup-uv@v7
      - run: uv pip install --system -e ".[all]" --group=docs
      - name: Configure git
        run: |
          git config user.name  "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
      - name: Deploy latest (main branch)
        if: github.ref == 'refs/heads/main'
        run: mike deploy --push --update-aliases dev latest
      - name: Deploy versioned (tag)
        if: startsWith(github.ref, 'refs/tags/')
        run: |
          VERSION=${GITHUB_REF#refs/tags/v}
          mike deploy --push --update-aliases "$VERSION" stable
          mike set-default --push stable
```

Notebook execution is **not** part of this workflow. Notebooks are
committed with their outputs (the standard scientific-Python workflow).
`mkdocs-jupyter` is configured with `execute: false`. Re-execution happens
manually before a release (see §8).

### `.github/workflows/publish.yml` — public PyPI via OIDC

PyPI trusted publishing replaces the cookiecutter's username/password
publish to the synthpop private index. No secrets stored in the repo.

```yaml
name: Publish
on:
  push: { tags: ['v*.*.*'] }
  workflow_dispatch:
permissions:
  id-token: write                                 # required for OIDC
  contents: read
jobs:
  publish:
    runs-on: ubuntu-latest
    environment:
      name: pypi
      url: https://pypi.org/p/mononet
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with: { python-version: "3.13" }
      - uses: astral-sh/setup-uv@v7
      - run: uv build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

One-time PyPI side setup: register the project on PyPI, then add
`davorrunje/mononet` + workflow `publish.yml` + environment `pypi` as a
trusted publisher. After that, every tag-pushed release goes to PyPI
without API tokens.

### `.github/workflows/bump-version.yml`

Same workflow as the cookiecutter, minus the `UV_INDEX_SYNTHPOP_PKGS_*`
env block.

### `.github/workflows/codeql.yml`

Kept as-is.

### Dependabot (`.github/dependabot.yml`)

Two ecosystems:
- `pip` (uv): groups updates by `lint`, `dev`, `docs`, runtime separately.
- `github-actions`: weekly.

Synthpop-specific dependabot config removed (it referenced the private
index).

## 7. License, NOTICE, and documentation

> **Superseded 2026-06-28:** the project relicensed from PolyForm
> Noncommercial 1.0.0 to **Apache-2.0**. The PolyForm details below are
> retained as historical record; the current license posture is defined in
> [`2026-06-28-relicense-apache-2.0-design.md`](2026-06-28-relicense-apache-2.0-design.md).

### `LICENSE` — verbatim PolyForm Noncommercial 1.0.0

The file holds the unmodified license text from
<https://polyformproject.org/licenses/noncommercial/1.0.0/>. No paraphrasing,
no clauses added inside — modifications void SPDX recognition. The
"Licensor" placeholder is filled with:

> **Licensor**: AIRT Technologies Ltd., Zagreb, Croatia.

### `NOTICE.md` — patent reservation + paper + commercial contact

```markdown
# NOTICE

`mononet` is licensed under the PolyForm Noncommercial License 1.0.0
(see `LICENSE`).

## Patent

This software implements technology covered by **U.S. Patent No.
11,551,063** ("Implementing monotonic constrained neural networks",
assignee: AIRT Technologies Ltd.). See
<https://patents.justia.com/patent/11551063>.

The PolyForm Noncommercial License covers your use of this *source code*
for noncommercial purposes. It does **not** grant any rights under the
patent, whether for commercial or noncommercial use. Practicing the
patented method (by any means, in any framework) requires a separate
patent license.

## Reference paper

Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic Neural
Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

If you use `mononet` in academic work, please cite this paper.

## Commercial licensing

For commercial use of the code and/or a license to U.S. Patent 11,551,063,
contact: **<licensing@airt.ai>** *(confirm address before first release)*.

## Trademarks

PyTorch is a trademark of the Linux Foundation. JAX is a trademark of
Google LLC. Keras is a trademark of Google LLC. Use of these trademarks
here indicates compatibility, not endorsement.
```

The `licensing@airt.ai` address is a placeholder to be confirmed before the
first PyPI release. Both `LICENSE` and `NOTICE.md` are referenced in
`pyproject.toml`'s `license-files` so they ship inside the wheel and sdist.

### `README.md` — rewrite

The cookiecutter README is mostly 1Password boilerplate. New structure:

```markdown
# mononet — Unconstrained Monotonic Neural Networks

[badges: PyPI version, supported Python, docs, build status, DOI]

Reference implementation of the unconstrained monotonic neural network
construction from:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic Neural
> Networks.* ICML 2023. https://arxiv.org/abs/2205.11775

First-class support for PyTorch, JAX (Flax NNX), and Keras 3.

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

## 60-second example

[Three short code blocks: one per backend, all training the same toy
monotonic regression and showing identical predictions.]

## License & patent

Code: PolyForm Noncommercial 1.0.0. Patent: US 11,551,063 reserved.
Commercial users contact <licensing@airt.ai>. See NOTICE.md.

## Documentation

Full docs at <https://davorrunje.github.io/mononet>.

## Citation

[BibTeX block for the paper.]
```

Everything else in the cookiecutter README (1Password walkthrough,
secret-configuration tables, internal release-process link) is deleted.
Devcontainer instructions move to `CONTRIBUTING.md`.

### `docs/` site

Top-level navigation in `mkdocs.yml`:

```
- Home: index.md                          (short pitch + install)
- Getting started:
    - PyTorch: guides/pytorch.md
    - JAX: guides/jax.md
    - Keras: guides/keras.md
- Concepts:
    - Monotonicity: concepts/monotonicity.md
    - Layer reference: concepts/layers.md
- Benchmarks (reproducing the paper):
    - Overview: benchmarks/index.md
    - <one .ipynb per experiment from the paper>
- API reference:
    - mononet.core: api/core.md          (mkdocstrings)
    - mononet.torch: api/torch.md
    - mononet.jax: api/jax.md
    - mononet.keras: api/keras.md
- About:
    - License & patent: about/license.md
    - Changelog: about/changelog.md      (literate-nav from CHANGELOG.md)
    - Citation: about/citation.md
```

`mkdocs-jupyter` config:

```yaml
plugins:
  - mkdocs-jupyter:
      execute: false                      # use committed notebook outputs
      include_source: true                # show download-as-.ipynb button
      ignore_h1_titles: true
```

`mike` is configured for versioned docs — landing page shows the latest
*stable* tag, `dev` alias tracks `main`.

### `CONTRIBUTING.md`

Rewritten to cover the new flow: devcontainer choice (CPU vs GPU flavor),
`uv sync`, pre-commit, how to run per-backend tests (`pytest tests/torch`),
how to re-execute benchmark notebooks before a release. Synthpop/Linear
sections deleted; a one-liner pointer to GitHub Issues replaces the Linear
workflow guide.

### `CHANGELOG.md`

First entry:

```
## [Unreleased]
### Added
- Initial release of mononet with PyTorch, JAX, and Keras backends.
- NumPy reference implementation and cross-backend equivalence test suite.
- Reproducing-the-paper benchmark notebooks.
```

### `SECURITY.md`

Kept as-is from the cookiecutter.

## 8. Release process (manual notebook execution)

Benchmarks can take hours (training MLPs on tabular data with multiple
seeds across hyperparameter grids). Running them on every docs deploy is
wasteful and brittle. The release process is:

1. Open a `gpu-*` devcontainer locally.
2. Run `./tools/execute-benchmarks.sh`:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   uv run jupyter nbconvert \
     --to notebook \
     --execute \
     --inplace \
     --ExecutePreprocessor.timeout=14400 \
     docs/docs/benchmarks/*.ipynb
   ```
3. `git diff docs/docs/benchmarks/` — sanity-check the new outputs.
4. Commit the notebook updates.
5. Run the `Bump Version` workflow → merge the version PR.
6. Tag the resulting commit `vX.Y.Z`. `publish.yml` ships to PyPI;
   `docs.yml` deploys the versioned docs (with the re-executed notebooks).

A `workflow_dispatch`-only `run-benchmarks.yml` is intentionally **not**
added in v0.x. Local execution is more debuggable; can be added later if
needed.

## 9. Migration plan (cookiecutter → design)

This is the ordered list of changes the implementation will perform. It is
not the implementation plan itself — that is produced by the
`writing-plans` skill in the next session. This is the migration map that
the design implies, so we can sanity-check no file is forgotten.

### Group A — strip cookiecutter's company-specific bits

1. Delete `.devcontainer/partner/`.
2. Delete `.devcontainer/default/initialize_devcontainer.sh`,
   `devcontainer.env*`, `post-start.sh`.
3. Remove `"initializeCommand"`, `"postStartCommand"`, `"secrets"` from
   `default/devcontainer.json`.
4. Delete `LINEAR_GUIDE.md`, `.linear.toml`,
   `.claude/skills/linear-cli/`.
5. Delete `codecov.yml`. Remove the Codecov upload step from `build.yml`.
6. Drop `[[tool.uv.index]] synthpop-pkgs` and `override-dependencies` from
   `pyproject.toml`.
7. Strip all `UV_INDEX_SYNTHPOP_PKGS_*` env blocks from every workflow.
8. Rewrite `LICENSE` (proprietary → PolyForm Noncommercial 1.0.0); add
   `NOTICE.md`.
9. Update copyright holder to **AIRT Technologies Ltd.** in `LICENSE` and
   in any source-file copyright headers.
10. Rewrite `README.md` (new structure from §7).
11. Rewrite `CONTRIBUTING.md` (drops Linear; new devcontainer flow).

### Group B — scaffold the package layout

12. Replace `mononet/__init__.py` (remove `HelloWorld`, set lazy public
    surface).
13. Create `mononet/core/{__init__.py,reference.py,config.py,types.py,numerics.py}`
    as stubs with `NotImplementedError` + docstrings citing the paper.
14. Create `mononet/torch/{__init__.py,_kernels.py,layers.py,models.py}`
    as stubs.
15. Create `mononet/jax/{__init__.py,_kernels.py,layers.py,models.py}`
    as stubs.
16. Create `mononet/keras/{__init__.py,_kernels.py,layers.py,models.py}`
    as stubs.
17. Add `mononet/py.typed` (empty file, PEP 561).
18. Create `tests/{core,torch,jax,keras,equivalence}/` with one trivial
    passing test each, so the per-backend CI matrix has something to run
    from day one.

### Group C — `pyproject.toml` & tool config

19. Update `[project]`: name, description, `requires-python = ">=3.11,<3.14"`,
    classifiers, URLs (arxiv paper + patent), license + license-files.
20. Drop `pydantic` from `dependencies`. Add `numpy`, `typing-extensions`
    (3.11 only).
21. Add `[project.optional-dependencies]`: `torch`, `jax`, `keras`, `all`,
    `torch-gpu`, `jax-gpu`, `keras-gpu`.
22. Rename `devdocs` → `docs` in `[dependency-groups]`. Add `bench` group.
    Add `hypothesis` to `dev`.
23. Drop `plugins = ["pydantic.mypy"]` and `[tool.pydantic-mypy]` from
    `[tool.mypy]`.
24. Update `[tool.ruff]` / `[tool.mypy]` targets to `py311` / `3.11`
    (effectively unchanged; verify).
25. Update `[tool.pytest.ini_options]` `addopts` (remove `--cov-report=xml`;
    keep terminal report).

### Group D — devcontainers

26. Replace `.devcontainer/default/devcontainer.json` & `docker-compose.yml`
    with cleaned versions (Python 3.13 base, no 1Password, no synthpop
    secrets).
27. Create `.devcontainer/gpu-torch/{Dockerfile,devcontainer.json,docker-compose.yml,setup.sh}`.
28. Create `.devcontainer/gpu-jax/{Dockerfile,devcontainer.json,docker-compose.yml,setup.sh}`.
29. Create `.devcontainer/gpu-keras/{Dockerfile,devcontainer.json,docker-compose.yml,setup.sh}`.
30. Consolidate `.devcontainer/shared/`
    (`install_common_tools.sh`, `install_uv.sh`, `setup_path.sh`,
    `post-create.sh`).

### Group E — CI workflows

31. Rewrite `.github/workflows/build.yml` (per-backend matrix from §6).
32. Rewrite `.github/workflows/publish.yml` (PyPI trusted publishing via
    OIDC; no secrets).
33. Update `.github/workflows/docs.yml` (mike-based versioned deploy; no
    notebook execution).
34. Update `.github/workflows/bump-version.yml` (strip synthpop env vars).
35. Keep `.github/workflows/codeql.yml` as-is.
36. Update `.github/dependabot.yml` (drop synthpop registry; group
    lint/dev/docs/runtime updates).

### Group F — docs site

37. Update `docs/mkdocs.yml`: new nav (§7), add `mkdocs-jupyter` plugin,
    add `mike` versioning.
38. Replace `docs/docs/index.md` with the new short pitch.
39. Create `docs/docs/guides/{pytorch,jax,keras}.md` stubs.
40. Create `docs/docs/concepts/{monotonicity,layers}.md` stubs.
41. Create `docs/docs/benchmarks/index.md` + a single placeholder notebook
    stub.
42. Create `docs/docs/api/{core,torch,jax,keras}.md` (each a
    `mkdocstrings: identifier: mononet.<name>` page).
43. Create `docs/docs/about/{license,changelog,citation}.md`.

### Group G — release tooling

44. Create `tools/execute-benchmarks.sh` (manual notebook re-execution).
45. Keep existing tools: `build-docs.sh`, `serve-docs.sh`, `lint.sh`,
    `static-analysis.sh`, `get-version.sh` (re-verify they still work
    after the synthpop strip).

### What is intentionally not in this migration

- **Actual algorithm implementation** — filling in the NumPy reference,
  the backend kernels, the equivalence tests with real cases, the
  benchmark notebooks. That is the work of the implementation plan
  (next session), one feature at a time, TDD-style.
- **First PyPI release** — happens after the implementation plan
  completes a usable v0.1.0.

### Order rationale

Groups A–C make the repo public-shaped — no functional changes. Group D
unblocks contributors. Group E unblocks PRs. Groups F–G complete the
public-facing pieces. After this whole migration, the repo has:

- a clean, license-correct, citable shell,
- working CI on three OSes × three Pythons × three backends with stub tests,
- a deployed docs site (placeholder content),
- devcontainers covering CPU + 3 GPU flavors,
- no AIRT-internal / synthpop / cookiecutter residue.

The implementation plan can then focus purely on the algorithm.

## 10. Open items to confirm before the first release

- `licensing@airt.ai` email address in `NOTICE.md` and `README.md`.
- AIRT Technologies Ltd. legal address line wording in `LICENSE`.
- Final BibTeX block for the citation page (paper venue/year cite key).
- PyPI project registration + trusted-publisher configuration
  (`davorrunje/mononet`, environment `pypi`).

None of these block the migration work; they only block the first tag.
