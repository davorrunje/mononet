# Multi-version type checking — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `mypy` once per supported Python version — each in an environment resolved for that version — with the version list derived from `requires-python`.

**Architecture:** A dependency-light script expands `[project] requires-python` into concrete versions. A shell primitive runs `mypy` either in the ambient environment (pre-commit hook, one version) or isolated per version via `UV_PROJECT_ENVIRONMENT` + `uv run --python`. CI consumes the same script to build a matrix, one job per version. `mypy`'s static `python_version` is deleted so the ambient run always matches the stubs installed alongside it.

**Tech Stack:** bash, uv (0.11.28), mypy 2.3.1, `packaging` via PEP 723 inline script metadata, GitHub Actions dynamic matrix (`fromJson`), pre-commit local hooks, pytest.

**Spec:** [`docs/superpowers/specs/2026-08-26-multi-version-typecheck-design.md`](../specs/2026-08-26-multi-version-typecheck-design.md)
**Issue:** [#167](https://github.com/davorrunje/mononet/issues/167)

## Global Constraints

- Supported versions come from `[project] requires-python` in `pyproject.toml`, currently `>=3.11,<3.15` → `3.11 3.12 3.13 3.14`. Never hardcode this list in a second place.
- Python 3.11+, line length 88 (ruff). Strict mypy. Type hints on every function.
- New `.py` files start with `# SPDX-License-Identifier: Apache-2.0` and use MyST field-list docstrings (`:param x:`, `:returns:`, `:raises X:`); types come from annotations, never `:type:`.
- `[tool.mypy] files = ["mononet", "tests", "benchmarks"]` — `tools/` is **not** type-checked, but `tests/` **is**, so test code must satisfy strict mypy.
- `[tool.ruff] include` **does** cover `tools/**/*.py`, so new scripts must pass ruff.
- Scope is mypy only. Do **not** touch `[tool.ruff] target-version` or the pyupgrade `--py313-plus` hook; those are [#168](https://github.com/davorrunje/mononet/issues/168).
- Per-version environments must **never** be the project's own `.venv`: `uv run --python 3.11` without `UV_PROJECT_ENVIRONMENT` rebuilds `.venv` on 3.11 and destroys the working environment.
- Commit on a branch, never on `main`. Commit with `--no-gpg-sign` and end messages with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## File Structure

| File | Responsibility |
|---|---|
| `tools/supported_pythons.py` | **create** — expand `requires-python` to a version list; `--json` for CI |
| `tests/tools/__init__.py` | **create** — make `tests/tools` a package, matching `tests/core/` |
| `tests/tools/test_supported_pythons.py` | **create** — tests for the expansion and the CLI |
| `tools/typecheck.sh` | **create** — run mypy: ambient (no arg) or isolated for one version |
| `tools/typecheck-all.sh` | **create** — sweep every supported version, report all failures |
| `tools/typecheck-pre-commit.sh` | **create** — hook wrapper, mirrors `tools/static-pre-commit.sh` |
| `tools/static-analysis.sh` | **modify** — drop mypy; keep bandit + semgrep |
| `pyproject.toml` | **modify** — delete `[tool.mypy] python_version` |
| `.pre-commit-config.yaml` | **modify** — add the `typecheck` local hook |
| `.github/workflows/build.yml` | **modify** — add `python-versions` + `typecheck` jobs; extend `check.needs` |
| `CONTRIBUTING.md` | **modify** — document the sweep; correct the "full gate" claim |

### Deviations from the spec (deliberate, both discovered while planning)

1. **Filename** is `tools/supported_pythons.py` (underscore), not `supported-pythons.py`. A hyphen makes the module unimportable, so tests could not exercise it directly. Matches the existing `tools/gen_versions_json.py`.
2. **Environment location** is `${XDG_CACHE_HOME:-$HOME/.cache}/mononet-typecheck/mypy-<v>`, not `.venvs/` inside the repo, so no `.gitignore` change is needed. Two reasons: the workspace is a **bind mount** in every devcontainer flavor (which is exactly why `.venv` is mounted as a container-private volume instead), so multi-GB environments there cost host disk and bind-mount I/O; and the cache directory shares a filesystem with `~/.cache/uv`, letting uv hardlink wheels instead of copying them. `MONONET_TYPECHECK_ENV_ROOT` overrides it.

---

### Task 1: Version list derived from `requires-python`

**Files:**
- Create: `tools/supported_pythons.py`
- Create: `tests/tools/__init__.py`
- Test: `tests/tools/test_supported_pythons.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `expand(specifier: str, candidates: Sequence[str] = CANDIDATES) -> list[str]`
  - `read_requires_python(pyproject: Path = DEFAULT_PYPROJECT) -> str`
  - `CANDIDATES: tuple[str, ...]`
  - CLI: `uv run --script tools/supported_pythons.py [--json] [--requires-python SPEC] [--pyproject PATH]`
  - Later tasks depend on the CLI only, in these two forms exactly:
    - `uv run --script tools/supported_pythons.py` → newline-separated (`3.11\n3.12\n3.13\n3.14`)
    - `uv run --script tools/supported_pythons.py --json` → `["3.11","3.12","3.13","3.14"]`

- [ ] **Step 1: Write the failing test**

Create `tests/tools/__init__.py` as an empty file (0 bytes), matching `tests/core/__init__.py`.

Create `tests/tools/test_supported_pythons.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Tests for `tools/supported_pythons.py`."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "supported_pythons.py"


def _load() -> ModuleType:
    """Import the script by path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("supported_pythons", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_bounded_range() -> None:
    assert _load().expand(">=3.11,<3.15") == ["3.11", "3.12", "3.13", "3.14"]


def test_expand_excludes_the_upper_bound() -> None:
    assert _load().expand(">=3.11,<3.12") == ["3.11"]


def test_expand_unbounded_runs_to_the_candidate_ceiling() -> None:
    module = _load()
    result = module.expand(">=3.13")
    assert result[0] == "3.13"
    assert result[-1] == module.CANDIDATES[-1]


def test_expand_honours_an_explicit_candidate_list() -> None:
    assert _load().expand(">=3.11,<3.15", ["3.10", "3.12", "3.99"]) == ["3.12"]


def test_read_requires_python_reads_the_project_table(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nrequires-python = ">=3.11,<3.13"\n',
        encoding="utf-8",
    )
    assert _load().read_requires_python(pyproject) == ">=3.11,<3.13"


def test_the_repo_requires_python_expands_to_a_sorted_nonempty_list() -> None:
    module = _load()
    versions = module.expand(module.read_requires_python())
    assert versions, "requires-python expanded to nothing"
    minors = [int(v.split(".")[1]) for v in versions]
    assert minors == sorted(minors)


def test_cli_emits_one_version_per_line() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--requires-python", ">=3.11,<3.14"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["3.11", "3.12", "3.13"]


def test_cli_json_is_parseable_by_the_ci_matrix() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--requires-python", ">=3.11,<3.13"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["3.11", "3.12"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/tools -v --no-cov`
Expected: FAIL — every test errors in `_load()` because `tools/supported_pythons.py` does not exist (`spec is not None` assertion fails, or `FileNotFoundError`).

- [ ] **Step 3: Write the script**

Create `tools/supported_pythons.py`:

```python
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24"]
# ///
"""Expand `requires-python` into the concrete Python versions mononet supports.

`[project] requires-python` in `pyproject.toml` is the single source of truth for
which Python versions this package supports. This script expands that specifier
so the local type-check sweep (`tools/typecheck-all.sh`) and the CI matrix
consume one answer instead of two hand-maintained lists.

The PEP 723 header above lets it run without syncing the project environment:

    uv run --script tools/supported_pythons.py           # 3.11 3.12 3.13 3.14
    uv run --script tools/supported_pythons.py --json    # ["3.11", ...]

:param --json: Emit a JSON array instead of one version per line. The CI matrix
    consumes this via `fromJson`.
:param --requires-python: Use this specifier instead of reading `pyproject.toml`.
:param --pyproject: Path to the `pyproject.toml` to read.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path

from packaging.specifiers import SpecifierSet

#: Candidate versions considered when expanding the specifier. The floor is
#: below anything this project ever supported; the ceiling only has to stay
#: ahead of CPython's release cadence.
CANDIDATES: tuple[str, ...] = tuple(f"3.{minor}" for minor in range(9, 21))

DEFAULT_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def expand(specifier: str, candidates: Sequence[str] = CANDIDATES) -> list[str]:
    """Return the candidates satisfying `specifier`, in ascending order.

    :param specifier: A PEP 440 version specifier, e.g. `">=3.11,<3.15"`.
    :param candidates: Versions to test. Defaults to `CANDIDATES`.
    :returns: The satisfying versions, e.g. `["3.11", "3.12", "3.13", "3.14"]`.
    """
    spec = SpecifierSet(specifier)
    return [version for version in candidates if spec.contains(version)]


def read_requires_python(pyproject: Path = DEFAULT_PYPROJECT) -> str:
    """Read `[project] requires-python` from a `pyproject.toml`.

    :param pyproject: Path to the file to read.
    :returns: The raw specifier string.
    :raises KeyError: If the `[project] requires-python` key is absent.
    :raises TypeError: If the key is present but not a string.
    """
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    requires = data["project"]["requires-python"]
    if not isinstance(requires, str):
        raise TypeError(
            f"{pyproject}: [project] requires-python must be a string, "
            f"got {type(requires).__name__}"
        )
    return requires


def main() -> None:
    """Print the supported versions, one per line or as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--requires-python", default=None)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    args = parser.parse_args()

    specifier = args.requires_python or read_requires_python(args.pyproject)
    versions = expand(specifier)
    if not versions:
        print(f"no candidate version satisfies {specifier!r}", file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps(versions) if args.as_json else "\n".join(versions))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/tools -v --no-cov`
Expected: PASS, 8 passed.

- [ ] **Step 5: Verify both CLI forms work without the project environment**

Run:
```bash
uv run --script tools/supported_pythons.py
uv run --script tools/supported_pythons.py --json
```
Expected: `3.11 3.12 3.13 3.14` one per line, then `["3.11", "3.12", "3.13", "3.14"]`.

- [ ] **Step 6: Lint**

Run: `uv run ruff check --exit-non-zero-on-fix && uv run ruff format --check && uv run mypy`
Expected: all three clean. (`tools/` is outside mypy's `files`; `tests/tools/` is inside it and must pass.)

- [ ] **Step 7: Correct the filename in the spec**

In `docs/superpowers/specs/2026-08-26-multi-version-typecheck-design.md`, replace both occurrences of `tools/supported-pythons.py` with `tools/supported_pythons.py`, and append to §1 after the sentence ending "`packaging` 26.3 is installed)":

```markdown
The script carries a PEP 723 inline metadata header declaring `packaging`, so
`uv run --script` executes it in a throwaway environment (~0.2 s) without
syncing the project — which is what lets the CI matrix job read the list before
any environment exists. The underscore in the filename is deliberate: a hyphen
would make the module unimportable and untestable.
```

- [ ] **Step 8: Commit**

```bash
git add tools/supported_pythons.py tests/tools/ \
  docs/superpowers/specs/2026-08-26-multi-version-typecheck-design.md
git commit --no-gpg-sign -m "$(cat <<'MSG'
feat(tools): derive the supported Python list from requires-python

Expands [project] requires-python into concrete versions so the local
type-check sweep and the CI matrix stop keeping separate hand-maintained
lists. A PEP 723 header declares packaging, so `uv run --script` runs it
in ~0.2s without syncing the project environment -- which is what lets a
CI job read the list before any environment exists.

Refs #167

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 2: The sweep scripts, and mypy loses its static target

**Files:**
- Create: `tools/typecheck.sh`, `tools/typecheck-all.sh`, `tools/typecheck-pre-commit.sh`
- Modify: `tools/static-analysis.sh` (remove the mypy block, lines 3–8)
- Modify: `pyproject.toml` (delete `[tool.mypy] python_version`, line 146)

**Interfaces:**
- Consumes: `uv run --script tools/supported_pythons.py` from Task 1 (newline form).
- Produces:
  - `tools/typecheck.sh` — no argument: mypy in the ambient environment. One argument (`3.11`): mypy isolated for that version. Exit status is mypy's.
  - `tools/typecheck-all.sh` — no arguments; runs every supported version, exits 1 listing the failures.
  - `tools/typecheck-pre-commit.sh` — no arguments; `cd`s to the repo root and calls `tools/typecheck.sh` with no argument. Task 3 wires this into `.pre-commit-config.yaml`.
  - Environment root: `${MONONET_TYPECHECK_ENV_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/mononet-typecheck}`.

- [ ] **Step 1: Write `tools/typecheck.sh`**

```bash
#!/bin/bash
# Run mypy, either in the ambient project environment or isolated for one
# supported Python version.
#
#   tools/typecheck.sh          # ambient env; what the pre-commit hook runs
#   tools/typecheck.sh 3.11     # isolated env resolved for 3.11
#
# The isolated form exists because mypy applies --python-version to the grammar
# it parses every file with, including dependency stubs, and the stubs installed
# differ per version: uv.lock resolves numpy 2.4.6 below 3.12 and 2.5.2 at or
# above it, and 2.5.2's stubs use PEP 695 `type` aliases that a 3.11 target
# cannot parse. A --python-version flag alone, in whatever env happens to be
# present, therefore cannot check the other versions.
set -e

cd "$(dirname "$0")"/..

version="${1:-}"

if [ -z "${version}" ]; then
  echo "Running mypy (ambient environment)..."
  exec uv run mypy
fi

# Never the project's own .venv: `uv run --python 3.11` would rebuild .venv on
# 3.11 and destroy the working environment. Kept out of the repo because the
# workspace is a bind mount in every devcontainer flavor, and kept next to the
# uv cache so wheels hardlink instead of copying.
env_root="${MONONET_TYPECHECK_ENV_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/mononet-typecheck}"

echo "Running mypy for Python ${version}..."
UV_PROJECT_ENVIRONMENT="${env_root}/mypy-${version}" \
  exec uv run --python "${version}" mypy --python-version "${version}"
```

- [ ] **Step 2: Verify both forms of `typecheck.sh`**

Run:
```bash
chmod +x tools/typecheck.sh
./tools/typecheck.sh
./tools/typecheck.sh 3.11
```
Expected: both print `Success: no issues found in 182 source files`. The second is slower (it creates an environment) and must **not** modify `.venv` — confirm with `.venv/bin/python --version`, which must still report 3.13.

- [ ] **Step 3: Write `tools/typecheck-all.sh`**

```bash
#!/bin/bash
# Type check every Python version in [project] requires-python.
#
# Runs all versions even after one fails: a sweep that stops at the first
# failure hides how much is broken. Exits non-zero naming the failures.
set -uo pipefail

cd "$(dirname "$0")"/..

mapfile -t versions < <(uv run --script tools/supported_pythons.py)

if [ ${#versions[@]} -eq 0 ]; then
  echo "no supported versions resolved; is [project] requires-python set?" >&2
  exit 1
fi

echo "Type checking ${#versions[@]} versions: ${versions[*]}"

failed=()
for version in "${versions[@]}"; do
  if ! ./tools/typecheck.sh "${version}"; then
    failed+=("${version}")
  fi
done

if [ ${#failed[@]} -gt 0 ]; then
  echo "" >&2
  echo "mypy FAILED for: ${failed[*]}" >&2
  exit 1
fi

echo ""
echo "mypy passed for: ${versions[*]}"
```

- [ ] **Step 4: Run the sweep**

Run: `chmod +x tools/typecheck-all.sh && ./tools/typecheck-all.sh`
Expected: four `Running mypy for Python 3.x...` blocks each ending `Success: no issues found in 182 source files`, then `mypy passed for: 3.11 3.12 3.13 3.14`, exit 0. Roughly 70 s on a warm uv cache, longer on the first run because interpreters are downloaded.

- [ ] **Step 5: Prove the sweep actually distinguishes versions**

This is the point of the whole change, so verify it rather than assuming. Append a 3.12-only stdlib symbol to a checked file:

```bash
cat >> mononet/core/numerics.py <<'EOF'


def _plan_probe() -> None:
    """Temporary: itertools.batched is 3.12+."""
    from itertools import batched

    list(batched([1, 2, 3], 2))
EOF
./tools/typecheck-all.sh; echo "exit=$?"
```

Expected: the 3.11 leg fails (`Module "itertools" has no attribute "batched"`), 3.12/3.13/3.14 pass, the summary reads `mypy FAILED for: 3.11`, and the exit status is 1.

Then revert and confirm green again:

```bash
git checkout -- mononet/core/numerics.py
./tools/typecheck-all.sh; echo "exit=$?"
```

Expected: `mypy passed for: 3.11 3.12 3.13 3.14`, exit 0. Do not commit the probe.

- [ ] **Step 6: Write `tools/typecheck-pre-commit.sh`**

Mirrors `tools/static-pre-commit.sh` exactly in shape:

```bash
#!/usr/bin/env bash

# A script for running mypy in the ambient environment,
# with all its dependencies installed.
#
# The pre-commit hook checks only this interpreter. CI sweeps every supported
# version (tools/typecheck-all.sh, and the `typecheck` matrix job).

set -o errexit

# Change directory to the project root directory.
cd "$(dirname "$0")"/..

./tools/typecheck.sh
```

- [ ] **Step 7: Remove mypy from `tools/static-analysis.sh`**

Delete these six lines (the file's current lines 3–8), leaving bandit and semgrep untouched:

```bash
echo "Running mypy..."
# benchmarks/ imports typer/optuna/etc., so mypy needs the `bench` group. It is
# in [tool.uv] default-groups for exactly that reason -- no flag needed here.
uv run mypy

```

The result must begin:

```bash
#!/bin/bash
set -e

echo "Running bandit..."
uv run bandit -c pyproject.toml -r mononet
```

- [ ] **Step 8: Delete mypy's static target**

In `pyproject.toml`, remove the line

```toml
python_version = "3.12"  # TODO: should cover all version
```

and insert in its place:

```toml
# No `python_version`: mypy would then apply it to the grammar it parses every
# file with, including dependency stubs, and the stubs installed differ per
# version (uv.lock resolves numpy 2.4.6 below 3.12, 2.5.2 at or above). Pinning
# any single value breaks a bare `uv run mypy` in some environment. Left unset,
# mypy targets the interpreter it runs under, which by construction matches the
# stubs beside it. Explicit targets come from tools/typecheck-all.sh, which
# pairs each --python-version with an environment resolved for it. See #167.
```

- [ ] **Step 9: Verify the split**

Run:
```bash
chmod +x tools/typecheck-pre-commit.sh
./tools/static-analysis.sh
./tools/typecheck-pre-commit.sh
```
Expected: the first prints only `Running bandit...` / `Running semgrep...` and no mypy output, ending `0 findings`; the second prints `Running mypy (ambient environment)...` then `Success: no issues found in 182 source files`.

- [ ] **Step 10: Commit**

```bash
git add tools/typecheck.sh tools/typecheck-all.sh tools/typecheck-pre-commit.sh \
  tools/static-analysis.sh pyproject.toml
git commit --no-gpg-sign -m "$(cat <<'MSG'
feat(tools): type check every supported Python version

tools/typecheck.sh runs mypy either in the ambient environment or
isolated for one version via UV_PROJECT_ENVIRONMENT + `uv run --python`;
tools/typecheck-all.sh sweeps every version from requires-python, runs
all of them even after a failure, and exits naming which failed.

Isolation is required, not tidiness. mypy applies --python-version to the
grammar it parses every file with, including dependency stubs, and the
stubs differ per version: uv.lock resolves numpy 2.4.6 below 3.12 and
2.5.2 at or above, and 2.5.2 uses PEP 695 `type` aliases a 3.11 target
cannot parse. Verified by making the 3.11 leg fail on itertools.batched
while the others passed.

mypy's static python_version is deleted rather than lowered: any single
pinned value breaks a bare `uv run mypy` in some environment. Unset, mypy
targets the interpreter it runs under, which matches the stubs beside it.
mypy also moves out of tools/static-analysis.sh, which keeps the
version-independent bandit and semgrep.

Refs #167

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 3: pre-commit hook

**Files:**
- Modify: `.pre-commit-config.yaml` (insert a hook after the `static-analysis` block, which ends at line 52)

**Interfaces:**
- Consumes: `tools/typecheck-pre-commit.sh` from Task 2.
- Produces: a hook with `id: typecheck`, runnable as `uv run pre-commit run typecheck --all-files`.

- [ ] **Step 1: Add the hook**

Insert immediately after the existing `static-analysis` hook block (before the `reference-hash` `- repo: local` block):

```yaml
-   repo: local
    hooks:
    -   id: typecheck
        name: Type check (this environment's Python)
        entry: "tools/typecheck-pre-commit.sh"
        language: python
        types: [python]
        require_serial: true
        verbose: true
```

The `stages` key is deliberately omitted so the hook inherits `default_stages: [pre-commit, pre-merge-commit]` **plus** the `manual` stage used by the CI `pre-commit` job — matching how `lint` and `static-analysis` are declared.

- [ ] **Step 2: Verify the hook runs and passes**

Run: `uv run pre-commit run typecheck --all-files`
Expected: `Type check (this environment's Python)....Passed`, with `Running mypy (ambient environment)...` and `Success: no issues found in 182 source files` in the verbose output.

- [ ] **Step 3: Verify the whole gate still passes**

Run: `uv run pre-commit run --all-files`
Expected: every hook passes. `Static analysis` no longer prints mypy output; `Type check` does.

- [ ] **Step 4: Verify the hook is genuinely narrower than CI**

Run:
```bash
uv run pre-commit run typecheck --all-files
uv run --script tools/supported_pythons.py
```
Expected: the hook reports one mypy run, while the script lists four versions. This asymmetry is the documented tradeoff, and Task 5 writes it down.

- [ ] **Step 5: Commit**

```bash
git add .pre-commit-config.yaml
git commit --no-gpg-sign -m "$(cat <<'MSG'
build: add a typecheck pre-commit hook

mypy moved out of tools/static-analysis.sh in the previous commit, so it
needs its own hook or commits stop being type checked at all. The hook
runs the ambient interpreter only (~5s); the four-version sweep runs in
CI, because paying ~70s and ~3GB on every commit is not worth it for an
error class a required check catches before merge.

Refs #167

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 4: CI matrix

**Files:**
- Modify: `.github/workflows/build.yml` — insert two jobs after `static-analysis` (which ends at line 31, before `pre-commit:` at line 33); extend `check.needs` (line 139)

**Interfaces:**
- Consumes: `uv run --script tools/supported_pythons.py --json` from Task 1; `tools/typecheck.sh <version>` from Task 2.
- Produces: status checks named `typecheck (3.11)` … `typecheck (3.14)`, required via `check`.

- [ ] **Step 1: Add the two jobs**

Insert between the `static-analysis` job and `pre-commit:`:

```yaml
  python-versions:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    outputs:
      list: ${{ steps.expand.outputs.list }}
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v9.0.0
      # PEP 723 metadata in the script means uv resolves `packaging` into a
      # throwaway environment; no setup-python and no project sync needed.
      - name: Expand requires-python into a version list
        id: expand
        run: |
          list="$(uv run --script tools/supported_pythons.py --json)"
          echo "resolved: ${list}"
          echo "list=${list}" >> "$GITHUB_OUTPUT"

  typecheck:
    needs: [python-versions]
    runs-on: ubuntu-latest
    permissions:
      contents: read
    strategy:
      fail-fast: false
      matrix:
        python-version: ${{ fromJson(needs.python-versions.outputs.list) }}
    steps:
      - uses: actions/checkout@v7
      - uses: astral-sh/setup-uv@v9.0.0
      # No setup-python: `uv run --python` in tools/typecheck.sh fetches the
      # interpreter, so the matrix value is the only place the version appears.
      - name: Type check
        run: ./tools/typecheck.sh ${{ matrix.python-version }}
```

`fail-fast: false` matches the existing `test` job: one broken version must not cancel the others, or the summary is useless.

- [ ] **Step 2: Require the new job**

Change `check`'s `needs` from

```yaml
    needs: [static-analysis, pre-commit, docs-smoke, test, coverage]
```

to

```yaml
    needs: [static-analysis, pre-commit, docs-smoke, test, coverage, typecheck]
```

`python-versions` is intentionally left out: if it fails, `typecheck` is *skipped*, and `alls-green` fails on a skipped needed job — so the failure still blocks the merge instead of passing silently.

- [ ] **Step 3: Validate the workflow locally**

Run: `uv run pre-commit run check-yaml --files .github/workflows/build.yml`
Expected: `check yaml...Passed`.

Then confirm the matrix expression consumes real output:

```bash
uv run --script tools/supported_pythons.py --json
```
Expected: `["3.11", "3.12", "3.13", "3.14"]` — valid JSON for `fromJson`, four matrix legs.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/build.yml
git commit --no-gpg-sign -m "$(cat <<'MSG'
ci: type check every supported Python version in a matrix

Adds a python-versions job that expands requires-python to JSON and a
typecheck matrix job that consumes it via fromJson, so adding a version
to requires-python extends CI with no workflow edit. Each version gets
its own status check, which names the failing version instead of hiding
it in a combined log.

Neither job needs setup-python: `uv run --python` fetches the interpreter
and the PEP 723 script resolves its own dependency. bandit and semgrep
stay outside the matrix, since running them four times gives four
identical results.

check.needs gains typecheck. python-versions is deliberately absent: if
it fails, typecheck is skipped, and alls-green fails on a skipped needed
job rather than passing silently.

Refs #167

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 5: Documentation

**Files:**
- Modify: `CONTRIBUTING.md` § "Lint, format, static analysis" (the command block and the paragraph beginning "`pre-commit` **is the full gate**")

**Interfaces:**
- Consumes: `tools/typecheck-all.sh` from Task 2.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Update the command block**

Replace

```bash
uv run mypy                                  # strict type check
```

with

```bash
uv run mypy                                  # strict type check, this env's Python
./tools/typecheck-all.sh                     # strict type check, every supported Python
```

- [ ] **Step 2: Correct the "full gate" claim**

This paragraph is now false: the hooks check one Python version and CI checks four. Replace the sentence

> so a clean `git commit` means the change already passes what CI enforces.

with

> so a clean `git commit` means the change already passes almost everything CI
> enforces. The exception is type checking across versions: the `typecheck` hook
> checks only the interpreter in your environment, while CI runs `mypy` against
> every version in `requires-python` (one status check each). Run
> `./tools/typecheck-all.sh` before pushing if you touched anything
> typing-sensitive — it takes about a minute warm, and the first run downloads an
> interpreter per version.

- [ ] **Step 3: Verify the docs still build and the prose passes the hooks**

Run: `uv run pre-commit run --files CONTRIBUTING.md`
Expected: all applicable hooks pass, including `codespell`.

- [ ] **Step 4: Full gate, one more time**

Run: `uv run pre-commit run --all-files && ./tools/typecheck-all.sh`
Expected: every hook passes, then `mypy passed for: 3.11 3.12 3.13 3.14`.

- [ ] **Step 5: Commit**

```bash
git add CONTRIBUTING.md
git commit --no-gpg-sign -m "$(cat <<'MSG'
docs: document the multi-version type check sweep

Adds ./tools/typecheck-all.sh to the static-analysis commands and
corrects the claim that a clean commit passes what CI enforces -- true
before, false now that the hook checks one interpreter and CI checks
every version in requires-python. Says which, and what to run before
pushing typing-sensitive changes.

Refs #167

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
MSG
)"
```

---

### Task 6: Open the PR

**Files:** none.

- [ ] **Step 1: Confirm the tree is green**

Run: `uv run pre-commit run --all-files && ./tools/typecheck-all.sh && uv run pytest tests/tools -v --no-cov`
Expected: all hooks pass, `mypy passed for: 3.11 3.12 3.13 3.14`, 8 tests pass.

- [ ] **Step 2: Push and open the PR**

Follow the **create-pr** skill (`.claude/skills/create-pr/SKILL.md`). The body must include `Closes #167` and note in the description that the checks list gains four `typecheck (3.x)` entries, so the first run is the real verification of Task 4.

- [ ] **Step 3: Verify in CI what cannot be verified locally**

After the run completes, confirm:
- four `typecheck (3.11 … 3.14)` statuses exist and all pass
- `python-versions` logged `resolved: ["3.11", "3.12", "3.13", "3.14"]`
- `static-analysis` no longer prints mypy output
- `check` passes with `typecheck` among its needs

---

## Self-review

**Spec coverage.** §1 version list → Task 1. §2 local sweep → Task 2 (Steps 1–5). §3 mypy loses its static target → Task 2 (Step 8). §4 static analysis splits → Task 2 (Step 7) + Task 3. §5 hook checks one version → Task 3 + Task 5. §6 CI → Task 4. "Known duplication" → Task 3 Step 3 observes it. "Unreleased versions in the range" → no task needed; the spec records it as intended behaviour, and Task 1's `test_expand_unbounded_runs_to_the_candidate_ceiling` pins the mechanism that produces it. Every spec verification bullet maps to a step: the sweep and its per-version naming (T2/S4), the deliberate 3.11-only error (T2/S5), the script's output for the current and a widened range (T1/S4 tests), the numpy fork (T2/S5 covers the mechanism; the 3.11 leg passing in T2/S4 is the regression case), pre-commit including the hook (T3/S2–3), a bare `uv run mypy` still passing (T2/S9), and CI's four statuses (T6/S3).

**Placeholders.** None: every code step carries complete file contents or an exact before/after, every command has expected output, and no step defers work.

**Name consistency.** `expand` / `read_requires_python` / `CANDIDATES` / `DEFAULT_PYPROJECT` are defined in Task 1 and used with those names in its tests. `tools/typecheck.sh` takes an optional positional version in Task 2 and is called that way in Task 4. `MONONET_TYPECHECK_ENV_ROOT` appears only in Task 2. The hook `id: typecheck` in Task 3 matches the `pre-commit run typecheck` commands. `python-versions` job outputs `list`, consumed as `needs.python-versions.outputs.list` in Task 4.
