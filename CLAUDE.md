# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo posture

- **The user of this repo is the paper's first author** (Davor Runje, ICML 2023, <https://arxiv.org/abs/2205.11775>). Default to a senior-collaborator tone; do not over-explain monotonic-network basics back at them.
- **License is Apache-2.0** (copyright holder: Davor Runje). Commercial and noncommercial use are both permitted. The underlying technique is described in **U.S. Patent 11,551,063** (assignee of record: AIRT Technologies Ltd., now winding down); the Apache-2.0 license (section 3) grants the patent rights needed to use this code. There is no noncommercial restriction. See [NOTICE.md](NOTICE.md).
- **No sycophantic openers or closing fluff.** Be terse in output, thorough in reasoning. Prefer editing over rewriting whole files. Test before declaring done. User instructions override this file.

## What this project is

A multi-backend implementation of the Constrained Monotonic Neural Network construction from the paper. First-class support for **PyTorch**, **JAX (Flax NNX)**, and **Keras 3** under a single installable package with optional extras. Distributed on PyPI as `mononet`. Repo: <https://github.com/davorrunje/mononet>.

The published wheel ships **layers only** — no training loops, no dataset loaders, no benchmark code. Benchmarks live in the repo (under `benchmarks/`, planned in Sub-project B) but are not part of the package.

## Key references

Source papers live under [docs/references/](docs/references/). PDFs are accompanied by a curated Markdown digest (accurate equations, repo-relevant notes) — read the digest first; consult the PDF when precision matters.

| Paper | Implements | Files |
|---|---|---|
| Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — arXiv:2205.11775 | The base CMFCL (`mode="mixed"`): `\|W\|_t` weight constraint. Paper uses a 3-class activation split `(s̆, ŝ, s̃)`; `mononet` uses a 2-class convex/concave split (see digest). | [PDF](docs/references/2205.11775v4.pdf) · [digest](docs/references/2205.11775-runje-2023-constrained-mnn.md) |
| Sartor et al., *Advancing Constrained Monotonic Neural Networks*, ICML 2025 — arXiv:2505.02537 | The activation switch (`mode="split"`): `f̂(x)=σ(W⁺x+b)−σ(W⁻x+b)`, no activation-split tuning. | [PDF](docs/references/2505.02537v2.pdf) · [digest](docs/references/2505.02537-sartor-2025-advancing-cmnn.md) |

## Workflow conventions

### Specs and plans live under `docs/superpowers/`

- [docs/superpowers/specs/](docs/superpowers/specs/) — design documents. Each one establishes the *what* and *why* for a discrete deliverable. **Always read the relevant spec before touching the code it covers.**
- [docs/superpowers/plans/](docs/superpowers/plans/) — implementation plans produced from specs by the `writing-plans` skill. Each plan is a stepwise checklist with review checkpoints.

The high-level project decomposition lives in five sub-project specs:

| Spec | Topic |
|---|---|
| [A](docs/superpowers/specs/2026-06-27-A-core-algorithm-and-backends-design.md) | Core algorithm, three backends, cross-backend equivalence |
| [B](docs/superpowers/specs/2026-05-22-B-paper-reproduction-design.md) | Reproduction of paper Tables 1 & 2 |
| [C](docs/superpowers/specs/2026-05-22-C-extended-benchmarks-design.md) | Extended datasets, ablations, scaling |
| [D](docs/superpowers/specs/2026-05-22-D-injective-monotonic-and-flows-design.md) | Strictly-monotonic primitives and normalizing flows |
| [E](docs/superpowers/specs/2026-05-22-E-lean-proofs-design.md) *(moved out)* | Lean 4 + mathlib4 formalization — now the standalone [neural-network-proofs](https://github.com/davorrunje/neural-network-proofs) repo |

Each sub-project spec sits below the parent meta-spec [2026-05-21-mononet-package-design.md](docs/superpowers/specs/2026-05-21-mononet-package-design.md), which establishes the package layout, license posture, CI matrix, and naming conventions. **Read the parent spec first** when you need the global picture.

> Stale reference in the parent meta-spec: it mentions **MkDocs**, but the project migrated to **Sphinx + myst-nb** ([2026-05-22-sphinx-migration-design.md](docs/superpowers/specs/2026-05-22-sphinx-migration-design.md)). The Sphinx-based docs are the current source of truth — do not propose MkDocs changes.

### Research workflow: defendable-science (science layer)

The scientific work runs on the **[defendable-science](https://github.com/davorrunje/defendable-science)** plugin (enabled in `.claude/settings.json`; CLI via `uv tool install defendable-science` — deliberately **not** in `pyproject.toml`). It is the *science* layer (hypotheses → papers → thesis, plus `literature`, `dataset`, `progress`, `defend`), governed by **agency** (the author makes and signs every material decision; AI drafts) and **understanding** (`defend`). Its git-native content lives under [`docs/research/`](docs/research/) (papers, hypotheses, thesis), [`datasets.yml`](datasets.yml), and [`.defendable-science/config.yml`](.defendable-science/config.yml); `docs/research/` is excluded from the Sphinx build.

defendable-science **delegates outward** via two bindings in `.defendable-science/config.yml`:

- `engineering_backend: superpowers` — code work (design/plan/implement) hands off to the superpowers flow; **`docs/superpowers/` is the engineering record** the science layer cites.
- `experiment_backend: benchmarks/` — the harness whose result JSONs `findings.md` cite as evidence (run-refs).

Setup + rationale: [2026-07-21-defendable-science-integration-design.md](docs/superpowers/specs/2026-07-21-defendable-science-integration-design.md). Current state: [`docs/research/dashboard.md`](docs/research/dashboard.md). When doing scientific work (proposing/testing a hypothesis, positioning a paper, registering a dataset), **use the defendable-science skills** and honor the named-sign-off requirement on verdicts.

**Research contact email:** `davor.runje@fer.hr` (the author's academic address) is the canonical email for all research-facing use — OpenAlex polite-pool `mailto`, Semantic Scholar / API registrations, paper author metadata, and correspondence. It is already the `authors`/`maintainers` email in `pyproject.toml` and the security contact in `SECURITY.md`; keep new research artifacts consistent with it. (This is distinct from the git commit identity.) **Secrets** (`S2_API_KEY`, etc.) live in the defendable-science key store at `.defendable-science/keys.json`, which is **gitignored — never commit it**; set keys via `defendable-science keys set <NAME>` (hidden prompt), and `defendable-science keys check` to verify presence.

### Follow-ups become GitHub issues

Whenever you defer work, note a follow-up, or find a problem you won't fix now, **create a GitHub issue for it** — don't leave it only in a spec's "Follow-ups" list, a PR comment, a code `TODO`, or conversational memory. Each issue must be **self-contained**: a future session has only the repository content and the issue text (not this conversation or PR thread), so include the context, the exact repo locations, and acceptance criteria needed to complete it cold. Use the **create-issue** skill for the standard format: [.claude/skills/create-issue/SKILL.md](.claude/skills/create-issue/SKILL.md) (template in [STYLE.md](.claude/skills/create-issue/STYLE.md)). That skill also defines the **closing** convention — a closed issue records *how* it was resolved (`Closes #NN` in the PR, or an explicit resolution comment); never close silently.

## Architecture

### Multi-backend pattern

Each backend mirrors the same internal shape so contributors can move between them:

```
mononet/<backend>/        # backend ∈ {torch, jax, keras}
├── _kernels.py           # private, pure-functional ops — the math, in framework-native tensors
└── layers.py             # public, Module/Layer wrappers around _kernels
```

- `_kernels.py` is **stateless**. Everything (weights, masks, splits) is passed in. This is what the equivalence harness validates.
- `layers.py` holds all public layer classes: `MonoLinear`/`MonoDense`, `MonoResidual`, and `MonoInput`. There are no composed model classes — users stack layers with the framework's native `Sequential` (or equivalent).

[mononet/core/reference.py](mononet/core/reference.py) holds the **NumPy reference implementation** — the arithmetic ground truth. Every backend kernel is asserted equivalent to it within fixed tolerance.

[mononet/core/types.py](mononet/core/types.py) and [mononet/core/config.py](mononet/core/config.py) hold the **shared types** (`MonotonicityMask`, `ActivationSpec`, `InitSpec`, `MonoConfig`, `MonoResidualConfig`). These are stdlib `dataclasses`, not Pydantic, with JSON round-trip for benchmark reproducibility. **Pydantic was deliberately rejected** to keep the wheel light and avoid Rust-binary conflicts with other ML libraries — do not reintroduce it.

### Naming

- PyTorch and JAX: `MonoLinear` (mirrors their `Linear`).
- Keras: `MonoDense` (mirrors its `Dense`).
- `MonoResidual` and `MonoInput` share one name across all three backends.
- There are no composed model classes (`MonoMLP`/`MonoFeatureBlock` were dropped).
- Pure-function NumPy reference uses `snake_case` (`monotonic_dense`, `monotonic_residual`) to flag it as the reference, not a layer.

### Lazy backend imports

`import mononet` does not import torch/jax/keras. Use `from mononet.torch import …` (or `.jax` / `.keras`) to access backend layers. **Preserve this**: do not move backend imports into the top-level `__init__.py`.

### Cross-backend equivalence tests

[tests/equivalence/](tests/equivalence/) parametrizes a battery of pre-generated `(shape, dtype, mode, convex_fraction, activation, seed)` cases as committed JSON in `tests/equivalence/cases/`. The same vectors run every CI build — no flaky seeds.

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
./tools/typecheck-all.sh                             # strict type check, every supported Python
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

## Commits

- **Commit proactively.** Don't wait to be asked — commit at sensible checkpoints (a coherent change, tests passing) as you normally would. This overrides any default "commit only when the user asks" behavior.
- **Never commit directly to `main`.** Branch first, then commit on the branch.

## Pull requests

PR conventions live in [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md) (gh-CLI usage, description-file workflow, replying to review comments via REST + resolving review threads via GraphQL). The **create-pr** skill encodes the branch/checks/commit/body ritual and the `Closes #NN` convention: [.claude/skills/create-pr/SKILL.md](.claude/skills/create-pr/SKILL.md) (template in [STYLE.md](.claude/skills/create-pr/STYLE.md)). Issues are tracked in this repo's GitHub Issues tab.
