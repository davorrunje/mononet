# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Approach

- Think before acting. Read existing files before writing code.
- Be concise in output but thorough in reasoning.
- Prefer editing over rewriting whole files.
- Do not re-read files you have already read unless the file may have changed.
- Test your code before declaring done.
- No sycophantic openers or closing fluff.
- Keep solutions simple and direct.
- User instructions always override this file.

## Project Overview

**Monotonic Neural Networks** — Implementation of unconstrained neural networks

Repository: `https://github.com/davorrunje/mononet`

## Common Commands

### Testing
```bash
uv run pytest                                        # run all tests
uv run pytest tests/path/to/test_file.py             # single file
uv run pytest tests/path/to/test_file.py::test_name  # single test
uv run pytest -m "not slow"                          # exclude slow tests
```

### Linting & Formatting
```bash
uv run ruff check --exit-non-zero-on-fix             # lint
uv run ruff format                                   # format
```

### Static Analysis
```bash
uv run mypy                                          # type check (strict)
uv run bandit -c pyproject.toml -r mononet  # security scan
uv run semgrep scan --config auto --error            # semgrep
```

### Pre-commit
```bash
uv run pre-commit run --all-files                    # run all hooks
```

### Documentation
Built with Sphinx + PyData Sphinx Theme. See `docs/conf.py`.

```bash
./tools/build-docs.sh                                # build docs (sphinx-build -W)
./tools/serve-docs.sh                                # live preview (sphinx-autobuild)
uv run sphinx-multiversion -W docs docs/_build/html  # build all versions (CI uses this)
```

### Dependencies
```bash
uv sync                                              # install / sync lockfile
```

## Architecture & Structure

```
mononet/
├── mononet/   # main package
├── tests/                           # test suite (mirrors package structure)
├── docs/                            # documentation (Sphinx)
├── tools/                           # dev scripts
├── .github/                         # workflows & PR/issue templates
└── pyproject.toml                   # project config
```

Key technologies: Python 3.11+, uv, pytest, ruff, mypy, bandit, semgrep, Sphinx + PyData Theme.

## Code Style

- Python 3.11+, line length 88 (ruff)
- MyST field-list docstrings: `:param x: ...` / `:returns: ...` / `:raises X: ...`. Types come from signature annotations, never `:type:`/`:rtype:`. Body text is MyST markdown.
- Strict mypy throughout
- Type hints on all functions and methods
- Pydantic for structured data; dataclasses for simple value objects
- Async-first where applicable (pytest-asyncio)

## Pull Request Workflow

- **PRs**: see [PULL_REQUEST_GUIDE.md](PULL_REQUEST_GUIDE.md)
- **Issues**: tracked in this repository's GitHub Issues tab.
