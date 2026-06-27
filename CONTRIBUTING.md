# Contributing to mononet

Thank you for your interest in contributing to mononet! This guide
covers the development workflow.

## License & patent reminder

`mononet` is distributed under the PolyForm Noncommercial License 1.0.0,
and the underlying technique is covered by U.S. Patent 11,551,063. By
contributing, you confirm that your contribution is your own work and
that you license it under the same terms. See [`NOTICE.md`](NOTICE.md)
for the full statement. For commercial use questions, contact
**licensing@airt.ai**.

## Development environments

The repo ships five devcontainer flavors. Pick the one matching your
hardware:

| Flavor          | When to use                                                      |
|-----------------|------------------------------------------------------------------|
| `default`       | CPU work: writing code, running unit tests, building docs.       |
| `gpu-torch`     | GPU benchmarks against the paper's PyTorch baseline.             |
| `gpu-jax`       | GPU work with JAX (Flax NNX).                                    |
| `gpu-keras`     | GPU work with Keras 3 (backed by JAX with CUDA 12 by default).   |
| `proofs`        | Reviewing the Lean 4 / mathlib4 formalization under `proofs/` (CPU, no ML extras). |

In VS Code, `Ctrl/Cmd+Shift+P` → `Dev Containers: Reopen in Container`,
then pick the flavor by name.

Outside devcontainers, you need Python ≥3.11, [uv](https://docs.astral.sh/uv/),
and git.

## Setup

```bash
git clone https://github.com/davorrunje/mononet.git
cd mononet
uv sync                            # install runtime + dev + docs + lint
uv run pre-commit install          # install git hooks
```

## Running tests

```bash
uv run pytest                      # full suite (skips backends not installed)
uv run pytest tests/core           # framework-agnostic tests only
uv run pytest tests/torch          # PyTorch-only tests
uv run pytest tests/jax            # JAX-only tests
uv run pytest tests/keras          # Keras-only tests
uv run pytest tests/equivalence    # cross-backend numerical equivalence
```

Set the active backend with `MONONET_TEST_BACKEND={torch|jax|keras}` when
running the equivalence suite to mirror what a single CI matrix cell
does.

## Lint, format, static analysis

```bash
uv run ruff check --exit-non-zero-on-fix    # lint
uv run ruff format                           # format
uv run mypy                                  # strict type check
uv run bandit -c pyproject.toml -r mononet   # security scan
uv run semgrep scan --config auto --error    # semgrep
uv run pre-commit run --all-files            # everything pre-commit runs
```

## Building docs

```bash
./tools/build-docs.sh              # one-shot build
./tools/serve-docs.sh              # live preview
```

Benchmark notebooks under `docs/docs/benchmarks/` are committed with
their outputs and are **not** re-executed during a docs build. To
re-execute them before a release, see "Release process" below.

## Release process

The full maintainer runbook — one-time Trusted-Publishing setup, the Bump
Version action, TestPyPI rehearsal, and the GitHub-Release-triggered publish —
lives in [`docs/releasing.md`](docs/releasing.md).

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`, `build:`.

## Pull requests

See [`PULL_REQUEST_GUIDE.md`](PULL_REQUEST_GUIDE.md) for repo-specific
PR conventions. New issues go to the project's GitHub Issues tab.

## Coding conventions

- Python 3.11+, line length 88 (ruff).
- MyST field-list docstrings on all public functions and classes (`:param x: ...`, `:returns: ...`, `:raises X: ...`). Types come from signature annotations, never `:type:`/`:rtype:`. See [the spec](docs/superpowers/specs/2026-05-22-myst-docstrings-design.md) for the canonical format.
- Strict mypy throughout. Type hints on every function and method.
- Stdlib `dataclasses` for simple value objects; avoid adding new
  runtime dependencies without discussion.
- Tests use `pytest`. Per-backend tests live under `tests/<backend>/`
  and use `pytest.importorskip("<framework>")` so they skip cleanly
  when the backend is not installed.

## Reporting security issues

See [`SECURITY.md`](SECURITY.md).
