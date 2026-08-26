# Multi-version type checking — design

**Date:** 2026-08-26
**Status:** approved (brainstorming)
**Owner:** Davor Runje
**Issue:** [#167](https://github.com/davorrunje/mononet/issues/167)

## Purpose

`mononet` supports four Python versions (`requires-python = ">=3.11,<3.15"`), but
types are checked against exactly one of them, in whatever interpreter the
devcontainer or the CI job happens to provide. This document specifies a sweep
that runs `mypy` once per supported version, each in an environment resolved for
that version.

## Background: how the single-version check broke

Widening `requires-python` in [#166](https://github.com/davorrunje/mononet/pull/166)
split `numpy` in `uv.lock`:

| Python | numpy |
|---|---|
| `< 3.12` | 2.4.6 |
| `>= 3.12` | 2.5.2 |

numpy 2.5.2's stubs use PEP 695 aliases (`type _Falsy = ...`). mypy applies
`python_version` to the grammar it parses **every** file with, including
dependency stubs, so the 3.13 devcontainer — which installs 2.5.2 — could not run
with the 3.11 target that `pyproject.toml` declared:

```
numpy/__init__.pyi:737: error: Type statement is only supported in Python 3.12
and greater  [syntax]
Found 1 error in 1 file (errors prevented further checking)
```

mypy 2.3.1 is the newest release, so there was no upgrade that avoided the
choice. #166 moved the target to 3.12 to land, which cleared the error and
silently stopped validating 3.11.

Two distinct gaps follow, and both are in scope:

1. **A single target version cannot validate the others.** `python_version` gates
   typeshed (which stdlib APIs exist) as well as the grammar.
2. **The dependency set itself differs per version.** The numpy fork above is one
   case; the `docs` group already pins `sphinx==9.0.4; python_version < '3.12'`
   against `9.1.0` otherwise. A type error appearing under only one resolution is
   invisible to a single run.

Gap 2 is what makes a cheap fix impossible. Passing `--python-version 3.11` inside
the 3.13 environment does not work: that environment still has numpy 2.5.2, whose
stubs are unparsable under a 3.11 target. Each version needs its own environment.

## Measurements

Taken in the `gpu-torch` devcontainer, warm uv cache, one environment per version
under `UV_PROJECT_ENVIRONMENT`:

| Version | Result | Wall clock | numpy resolved |
|---|---|---|---|
| 3.11 | `Success: no issues found in 182 source files` | 19.6 s | 2.4.6 |
| 3.12 | `Success: no issues found in 182 source files` | 10.2 s | 2.5.2 |
| 3.13 | `Success: no issues found in 182 source files` | 15.2 s | 2.5.2 |
| 3.14 | `Success: no issues found in 182 source files` | 24.3 s | 2.5.2 |

Four environments totalled 2.9 GB. **All four versions pass today**, so this
change lands green rather than uncovering a backlog of type errors. The 3.11 row
also confirms the mechanism: that environment resolved numpy 2.4.6, so the sweep
genuinely exercises the dependency fork.

## Scope

**In scope:** running mypy against every supported Python version, locally and in
CI, from a single source of truth for the version list.

**Out of scope:** the other two settings that encode a target version and
currently disagree with `requires-python` — `[tool.ruff] target-version = "py311"`
and the pyupgrade hook's `--py313-plus`, which can rewrite source into syntax that
does not parse on 3.11 or 3.12. Filed as
[#168](https://github.com/davorrunje/mononet/issues/168).

## Design

### 1. Version list — derived, not duplicated

`tools/supported_pythons.py` reads `[project] requires-python` from
`pyproject.toml` with `tomllib`, expands it against candidate versions 3.9–3.20
using `packaging.specifiers.SpecifierSet`, and prints one version per line, or a
JSON array with `--json` for the CI matrix. Both dependencies are already
available (`tomllib` is stdlib on 3.11+; `packaging` 26.3 is installed).

The script carries a PEP 723 inline metadata header declaring `packaging`, so
`uv run --script` executes it in a throwaway environment (~0.2 s) without
syncing the project — which is what lets the CI matrix job read the list before
any environment exists. The underscore in the filename is deliberate: a hyphen
would make the module unimportable and untestable.

Adding 3.15 to `requires-python` then extends the local sweep and the CI matrix
with no second edit. This is the issue's central acceptance criterion.

### 2. Local sweep

`tools/typecheck.sh [version]` is the shared primitive:

- **no argument** — run `mypy` in the ambient project environment. This is what
  the pre-commit hook calls.
- **with a version** — run it isolated:

  ```bash
  UV_PROJECT_ENVIRONMENT=".venvs/mypy-$v" uv run --python "$v" mypy --python-version "$v"
  ```

`UV_PROJECT_ENVIRONMENT` is load-bearing, not hygiene. Without it,
`uv run --python 3.11` rebuilds the project's own `.venv` on 3.11 and destroys the
working environment. `.venvs/` is added to `.gitignore`.

`tools/typecheck-all.sh` loops every supported version, runs **all** of them even
after one fails, and exits non-zero naming which versions failed — a partial
sweep that stops at the first failure hides how much is broken.

**Unreleased versions in the range.** Because the list is derived from
`requires-python`, widening the range to a version that does not exist yet puts
it in the sweep, where `uv run --python` fails to find an interpreter. This is
the correct behaviour and is left as-is: it is the same signal
`actions/setup-python` gave in #166 when `3.15` was added to the test matrix
before release ("The version '3.15' ... was not found"). The failure names the
version, so the cause is not obscure.

### 3. mypy loses its static target

`[tool.mypy] python_version` is **removed**, not lowered. Setting it to the lowest
supported version is what broke: a bare `uv run mypy` in the 3.13 devcontainer
then meets numpy 2.5.2's stubs under a 3.11 target and dies. With the setting
absent, mypy targets the interpreter it runs under, which by construction matches
the stubs installed in that environment. Every explicit target then comes from the
sweep, which pairs each `--python-version` with an environment resolved for it.

### 4. Static analysis splits

`tools/static-analysis.sh` currently runs mypy, bandit and semgrep, and is invoked
both by CI and by the pre-commit hook. mypy moves out; the script keeps bandit and
semgrep, which are version-independent. Typing gets its own pre-commit hook,
`typecheck`, wired through `tools/typecheck-pre-commit.sh` — the same wrapper
pattern as the existing `tools/static-pre-commit.sh`.

### 5. The hook checks one version; CI checks all four

The hook runs the ambient version only (~5 s, today's speed). The sweep costs
~70 s warm and ~3 GB of environments, which is not worth paying on every commit
for an error class that a required check catches before merge.

The consequence is explicit and belongs in the docs rather than in a surprise: a
version-specific type error is caught at PR time, not commit time.
`CONTRIBUTING.md` documents `./tools/typecheck-all.sh` and states that the hook is
narrower than CI.

### 6. CI

| Job | Change |
|---|---|
| `python-versions` | **new** — emits `outputs.list` from `supported-pythons.py --json` |
| `typecheck` | **new** — `needs: [python-versions]`, `matrix.python-version: ${{ fromJson(...) }}`, runs `tools/typecheck.sh $v`. Needs only checkout + `setup-uv`; no `setup-python`, since `uv run --python` fetches the interpreter |
| `static-analysis` | unchanged in shape; its script no longer runs mypy |
| `check` | `needs` gains `typecheck`, so a failing version blocks merge |

bandit and semgrep stay out of the matrix deliberately — matrixing the whole
`static-analysis` job would run them four times for identical results.

## Known duplication

CI's `pre-commit` job runs hooks with `--hook-stage manual --all-files`, so the new
`typecheck` hook adds one ambient mypy (~15 s) that overlaps the matrix's 3.13 leg.
This is accepted: the alternative is excluding the hook from that job, which would
mean the hook set CI verifies no longer matches the hook set developers run.

## Verification

- [ ] `./tools/typecheck-all.sh` passes for 3.11, 3.12, 3.13 and 3.14, and its
      output names each version
- [ ] introducing a deliberate 3.11-only type error (e.g. a stdlib symbol added in
      3.12) fails the 3.11 leg and only that leg; reverted afterwards
- [ ] `tools/supported_pythons.py` emits `3.11 3.12 3.13 3.14` for the current
      `requires-python`, and emits five entries if it is temporarily widened to
      `<3.16`
- [ ] the 3.11 environment resolves numpy 2.4.6 and the others 2.5.2 — the
      regression case from #166
- [ ] `uv run pre-commit run --all-files` passes and the `typecheck` hook appears
      in its output
- [ ] a bare `uv run mypy` still passes in the devcontainer with
      `[tool.mypy] python_version` removed
- [ ] CI shows four `typecheck (3.x)` statuses and `check` requires them

## Follow-ups

- [#168](https://github.com/davorrunje/mononet/issues/168) —
  `[tool.ruff] target-version` and the pyupgrade `--py313-plus` hook disagree with
  `requires-python`; pyupgrade can rewrite source into syntax that does not parse
  on supported versions.
- `.github/workflows/build.yml`'s `test` job still hardcodes its
  `python-version` list. Once `python-versions` exists it could consume the same
  output, removing the last duplicate list. Deliberately not done here to keep
  this change to typing.
