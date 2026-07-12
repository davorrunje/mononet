# Contributing to mononet

Thank you for your interest in contributing to mononet! This guide
covers the development workflow.

## License of contributions

`mononet` is licensed under the Apache License 2.0. Contributions are
accepted inbound=outbound under section 5 of that license: unless you state
otherwise, any contribution you intentionally submit for inclusion is
licensed under Apache-2.0, with no additional terms. No CLA is required. See
[`NOTICE.md`](https://github.com/davorrunje/mononet/blob/main/NOTICE.md).

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

The devcontainers also **auto-provision your tooling** on build: pre-commit
hooks are installed, Claude Code plugins are set up, and this repo's Claude
sessions are shared with your host (see [Claude Code](#claude-code-plugins--sessions)).
Working locally, you do those steps yourself.

> **The devcontainer `.venv` is container-private.** Each flavor mounts the
> project virtualenv (`/workspaces/mononet/.venv`) as its own named Docker
> volume, isolated from any `.venv` on your host — so a host-side `uv` run and
> the container never clobber each other's environment (their interpreters live
> at different paths and can't be shared). It persists across rebuilds; if it
> goes stale, reset it with `docker volume rm <compose-project>_mononet-venv`
> (find the name via `docker volume ls | grep mononet-venv`) and rebuild.

## Claude Code (plugins & sessions)

The Claude Code plugins this repo uses are declared in
[`.devcontainer/claude-plugins.txt`](https://github.com/davorrunje/mononet/blob/main/.devcontainer/claude-plugins.txt) and
installed by `.devcontainer/shared/provision-claude-plugins.sh` (idempotent,
user scope). That one script is the source of truth for both environments:

- **In a devcontainer** — nothing to do. `post-create` runs the script
  (plugins) and installs the pre-commit hooks, and this repo's session
  transcripts are bind-mounted to/from your host, so a conversation started on
  the host continues in the container and vice-versa. You log in inside the
  container (auth is not shared).
- **Locally (host)** — run it once:

  ```bash
  bash .devcontainer/shared/provision-claude-plugins.sh   # install the repo's Claude plugins
  ```

  Your host `~/.claude` (settings, auth, other plugins) stays otherwise
  independent from the container's.

To add a plugin, append a `<marketplace-source>  <plugin@marketplace>` line to
`.devcontainer/claude-plugins.txt`, then re-run the script (host) or rebuild the
container.

> Host and container keep **independent** plugin/config state; only *this
> repo's* sessions are shared. Avoid running Claude Code on the host and in the
> container for this project **at the same time** — concurrent writes can
> interleave the shared transcript.

## Setup

```bash
git clone https://github.com/davorrunje/mononet.git
cd mononet
uv sync                            # install runtime + dev + docs + lint
uv run pre-commit install          # install git hooks (devcontainers do this for you)
```

If you skip the hooks, run the checks manually before pushing (see
[Lint, format, static analysis](#lint-format-static-analysis)).

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

`pre-commit` **is the full gate**: on every commit it runs all of the above
plus the docs build, codespell, secret detection, and file-hygiene hooks — so a
clean `git commit` means the change already passes what CI enforces. The same
checks run on demand whether or not the hooks are installed: `uv run pre-commit
run --all-files` for the lot, or the individual commands above piecemeal.

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
lives in [`docs/about/releasing.md`](https://github.com/davorrunje/mononet/blob/main/docs/about/releasing.md).

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/):
`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `ci:`, `build:`.

## Pull requests

See [`PULL_REQUEST_GUIDE.md`](https://github.com/davorrunje/mononet/blob/main/PULL_REQUEST_GUIDE.md) for repo-specific
PR conventions. New issues go to the project's GitHub Issues tab.

## Coding conventions

- Python 3.11+, line length 88 (ruff).
- MyST field-list docstrings on all public functions and classes (`:param x: ...`, `:returns: ...`, `:raises X: ...`). Types come from signature annotations, never `:type:`/`:rtype:`. See [the spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-05-22-myst-docstrings-design.md) for the canonical format.
- Strict mypy throughout. Type hints on every function and method.
- Stdlib `dataclasses` for simple value objects; avoid adding new
  runtime dependencies without discussion.
- Tests use `pytest`. Per-backend tests live under `tests/<backend>/`
  and use `pytest.importorskip("<framework>")` so they skip cleanly
  when the backend is not installed.

## Reporting security issues

See [`SECURITY.md`](https://github.com/davorrunje/mononet/blob/main/SECURITY.md).
