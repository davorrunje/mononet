# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo posture

- **The user of this repo is the paper's first author** (Davor Runje, ICML 2023, <https://arxiv.org/abs/2205.11775>). Default to a senior-collaborator tone; do not over-explain monotonic-network basics back at them.
- **License is PolyForm Noncommercial 1.0.0**, and the underlying technique is covered by **U.S. Patent 11,551,063** (assignee: AIRT Technologies Ltd.). Treat the noncommercial scope as a hard constraint: do not propose features whose primary purpose is helping commercial deployments, do not suggest copying ourselves into permissively-licensed repos, and route any "can I use this commercially?" question to **licensing@airt.ai**. See [NOTICE.md](NOTICE.md).
- **No sycophantic openers or closing fluff.** Be terse in output, thorough in reasoning. Prefer editing over rewriting whole files. Test before declaring done. User instructions override this file.

## What this project is

A multi-backend implementation of the Constrained Monotonic Neural Network construction from the paper. First-class support for **PyTorch**, **JAX (Flax NNX)**, and **Keras 3** under a single installable package with optional extras. Distributed on PyPI as `mononet`. Repo: <https://github.com/davorrunje/mononet>.

The published wheel ships **layers only** — no training loops, no dataset loaders, no benchmark code. Benchmarks live in the repo (under `benchmarks/`, planned in Sub-project B) but are not part of the package.

## Workflow conventions

### Specs and plans live under `docs/superpowers/`

- [docs/superpowers/specs/](docs/superpowers/specs/) — design documents. Each one establishes the *what* and *why* for a discrete deliverable. **Always read the relevant spec before touching the code it covers.**
- [docs/superpowers/plans/](docs/superpowers/plans/) — implementation plans produced from specs by the `writing-plans` skill. Each plan is a stepwise checklist with review checkpoints.

The high-level project decomposition lives in five sub-project specs dated 2026-05-22:

| Spec | Topic |
|---|---|
| [A](docs/superpowers/specs/2026-05-22-A-core-algorithm-and-backends-design.md) | Core algorithm, three backends, cross-backend equivalence |
| [B](docs/superpowers/specs/2026-05-22-B-paper-reproduction-design.md) | Reproduction of paper Tables 1 & 2 |
| [C](docs/superpowers/specs/2026-05-22-C-extended-benchmarks-design.md) | Extended datasets, ablations, scaling |
| [D](docs/superpowers/specs/2026-05-22-D-injective-monotonic-and-flows-design.md) | Strictly-monotonic primitives and normalizing flows |
| [E](docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md) | Lean 4 + mathlib4 formalization of paper theorems |

Each sub-project spec sits below the parent meta-spec [2026-05-21-mononet-package-design.md](docs/superpowers/specs/2026-05-21-mononet-package-design.md), which establishes the package layout, license posture, CI matrix, and naming conventions. **Read the parent spec first** when you need the global picture.

> Stale reference in the parent meta-spec: it mentions **MkDocs**, but the project migrated to **Sphinx + myst-nb** ([2026-05-22-sphinx-migration-design.md](docs/superpowers/specs/2026-05-22-sphinx-migration-design.md)). The Sphinx-based docs are the current source of truth — do not propose MkDocs changes.

## Architecture

### Multi-backend pattern

Each backend mirrors the same internal shape so contributors can move between them:

```
mononet/<backend>/        # backend ∈ {torch, jax, keras}
├── _kernels.py           # private, pure-functional ops — the math, in framework-native tensors
├── layers.py             # public, Module/Layer wrappers around _kernels
└── models.py             # public, composed models (MonoMLP, MonoFeatureBlock)
```

- `_kernels.py` is **stateless**. Everything (weights, masks, splits) is passed in. This is what the equivalence harness validates.
- `layers.py` wraps a kernel in the framework-idiomatic stateful container (`nn.Module`, `nnx.Module`, `keras.Layer`).
- `models.py` composes layers into the two architectures from the paper (Fig. 4 → `MonoMLP`, Fig. 5 → `MonoFeatureBlock`).

[mononet/core/reference.py](mononet/core/reference.py) holds the **NumPy reference implementation** — the arithmetic ground truth. Every backend kernel is asserted equivalent to it within fixed tolerance. Currently stubbed with `NotImplementedError`; signatures are locked.

[mononet/core/types.py](mononet/core/types.py) and [mononet/core/config.py](mononet/core/config.py) hold the **shared types** (`MonotonicityMask`, `ActivationSpec`, `InitSpec`, `MonoLinearConfig`). These are stdlib `dataclasses`, not Pydantic, with JSON round-trip for benchmark reproducibility. **Pydantic was deliberately rejected** to keep the wheel light and avoid Rust-binary conflicts with other ML libraries — do not reintroduce it.

### Naming

- PyTorch and JAX: `MonoLinear` (mirrors their `Linear`).
- Keras: `MonoDense` (mirrors its `Dense`).
- Composed models share `MonoMLP` / `MonoFeatureBlock` across backends.
- Pure-function NumPy reference uses `snake_case` (`monotonic_dense`, `monotonic_mlp`) to flag it as the reference, not a layer.

### Lazy backend imports

`import mononet` does not import torch/jax/keras. Use `from mononet.torch import …` (or `.jax` / `.keras`) to access backend layers. **Preserve this**: do not move backend imports into the top-level `__init__.py`.

### Cross-backend equivalence tests

[tests/equivalence/](tests/equivalence/) parametrizes a battery of pre-generated `(shape, dtype, mask, activation, split, seed)` cases as committed JSON in `tests/equivalence/cases/`. The same vectors run every CI build — no flaky seeds.

CI selects the active backend with `MONONET_TEST_BACKEND={torch|jax|keras}` and uses `pytest.importorskip` to skip the others. Locally:

```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/equivalence
```

### Devcontainer flavors

Four devcontainer flavors under [.devcontainer/](.devcontainer/):

| Flavor | Use |
|---|---|
| `default` | CPU work: code, unit tests, docs |
| `gpu-torch` | GPU benchmarks with PyTorch |
| `gpu-jax` | GPU work with JAX (Flax NNX) |
| `gpu-keras` | GPU work with Keras 3 (JAX backend + CUDA 12 by default) |

`shared/` holds scripts reused across flavors. Selection guidance is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Common commands

```bash
uv sync                                              # install / sync lockfile
uv run pytest                                        # full suite (skips uninstalled backends)
uv run pytest tests/path/to/test_file.py::test_name  # single test
uv run pytest -m "not slow"                          # exclude slow tests
uv run ruff check --exit-non-zero-on-fix             # lint
uv run ruff format                                   # format
uv run mypy                                          # strict type check
uv run pre-commit run --all-files                    # all hooks
./tools/build-docs.sh                                # one-shot Sphinx docs build
./tools/serve-docs.sh                                # live preview
./tools/execute-benchmarks.sh                        # re-execute benchmark notebooks (release prep)
```

Full reference (including per-backend test invocations, security/static-analysis tooling, and the release dance) is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Code style

- Python 3.11+, line length 88 (ruff).
- **MyST field-list docstrings** on all public functions and classes: `:param x: …`, `:returns: …`, `:raises X: …`. Types come from signature annotations, never `:type:` / `:rtype:`. Body text is MyST markdown. Canonical format spec: [2026-05-22-myst-docstrings-design.md](docs/superpowers/specs/2026-05-22-myst-docstrings-design.md).
- Strict mypy throughout. Type hints on every function and method.
- Stdlib `dataclasses` for simple value objects. Do not reintroduce Pydantic (see "Architecture" above).
- Async-first where applicable (pytest-asyncio in `dev` group).

## Pull requests

PR conventions live in [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md) (gh-CLI usage, description-file workflow, replying to review comments via REST + resolving review threads via GraphQL). Issues are tracked in this repo's GitHub Issues tab.
