# PyPI Alpha Release Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `mononet 0.0.0a0` to PyPI via a GitHub-Release-triggered publish workflow, with a manual version-bump action and a TestPyPI rehearsal path.

**Architecture:** Four independent changes — bump the package version; extend the manual `bump-version.yml` choices; rewrite `publish.yml` triggers/target/guard; add a `docs/releasing.md` runbook and reconcile `CONTRIBUTING.md`. No application code changes; "tests" are concrete verification commands (version readout, `uv build` artifact names, YAML validity, guard-logic shell test, strict docs build).

**Tech Stack:** GitHub Actions, `uv` (build + `uv version`), PyPI Trusted Publishing (OIDC), Sphinx (myst-nb).

## Global Constraints

- Package version target: **`0.0.0a0`** (PEP 440 normalized; wheel `mononet-0.0.0a0-py3-none-any.whl`, sdist `mononet-0.0.0a0.tar.gz`).
- Version source of truth is `pyproject.toml`; the git tag must match it on release events.
- Tag convention: **`v`-prefixed** (e.g. `v0.0.0a0`); the guard strips the leading `v` before comparing.
- Auth: **Trusted Publishing (OIDC)** on both PyPI and TestPyPI. **No API tokens, no secrets** added to any workflow.
- Publish triggers: `release: types: [published]` **and** `workflow_dispatch` (target-selectable). The bare `push: tags: v*.*.*` trigger is **removed**.
- Release events always target **real PyPI**; TestPyPI is reachable only via manual `workflow_dispatch`.
- Bump options: `patch, minor, major, alpha, beta, rc, stable` (default `patch`).
- `uv` pre-release semantics (uv 0.11.x): from a **prerelease** version, `--bump alpha|beta|rc|stable` works standalone (`0.0.0a0 → 0.0.0a1`, `→ 0.0.0`). From a **stable** version, `--bump alpha|beta|rc` errors and requires a release component too (`--bump patch --bump alpha`). This constraint is documented, not worked around.
- Publish does **not** re-run the test matrix (releases are cut from green `main`).
- Commits: branch is `feat/pypi-alpha-release` (already created). Never commit to `main`. All commits signed (Secretive SSH); end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- All commands run from the repo root `/Users/davor/Projects/PhD/mononet`.

---

### Task 1: Set package version to `0.0.0a0`

**Files:**
- Modify: `pyproject.toml:8`

**Interfaces:**
- Consumes: nothing.
- Produces: `pyproject.toml` version `0.0.0a0`, readable by `tools/get-version.sh` and `uv version --short`; consumed by `publish.yml` artifact verification (Task 3).

- [ ] **Step 1: Confirm the current version**

Run: `uv version --short`
Expected output: `0.0.0`

- [ ] **Step 2: Set the version**

Run: `uv version 0.0.0a0 --no-sync`
Expected output (last line): `mononet 0.0.0 => 0.0.0a0`

This rewrites `pyproject.toml:8` to:

```toml
version = "0.0.0a0"
```

- [ ] **Step 3: Verify the version readout**

Run: `uv version --short`
Expected output: `0.0.0a0`

- [ ] **Step 4: Verify build artifact names**

Run: `uv build`
Expected: command succeeds and `ls dist/` shows both:

```
mononet-0.0.0a0-py3-none-any.whl
mononet-0.0.0a0.tar.gz
```

(The `dist/` directory is git-ignored; do not commit it.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
build: set version to 0.0.0a0 for first alpha release

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Extend `bump-version.yml` with pre-release bump options

**Files:**
- Modify: `.github/workflows/bump-version.yml:11-14`

**Interfaces:**
- Consumes: nothing.
- Produces: a manual `Bump Version` workflow whose `bump_type` choice includes `alpha`/`beta`/`rc`/`stable`; runs `uv version --bump <type>` and opens a PR (unchanged downstream).

- [ ] **Step 1: Replace the options list**

In `.github/workflows/bump-version.yml`, the current input block is:

```yaml
      bump_type:
        description: 'Version bump type'
        required: true
        default: 'patch'
        type: choice
        options:
          - patch
          - minor
          - major
```

Replace the `options:` list (and add the clarifying comment) so the block reads:

```yaml
      bump_type:
        description: 'Version bump type'
        required: true
        default: 'patch'
        type: choice
        # alpha/beta/rc/stable bump the *current* version. They work standalone
        # only when the current version is already a pre-release (e.g.
        # 0.0.0a0 -> alpha -> 0.0.0a1, or -> stable -> 0.0.0). From a stable
        # version, `uv version --bump alpha` errors and you must bump a release
        # component first (use patch/minor/major).
        options:
          - patch
          - minor
          - major
          - alpha
          - beta
          - rc
          - stable
```

Leave the rest of the file (the `bump-version` job, `uv version --bump`, PR creation) unchanged.

- [ ] **Step 2: Verify the YAML is valid and the options are present**

Run:

```bash
uv run python - <<'PY'
import yaml
d = yaml.safe_load(open(".github/workflows/bump-version.yml"))
# `on` parses to True (YAML 1.1 boolean); fetch the trigger block by that key.
opts = d[True]["workflow_dispatch"]["inputs"]["bump_type"]["options"]
assert opts == ["patch","minor","major","alpha","beta","rc","stable"], opts
print("bump options OK:", opts)
PY
```

Expected output: `bump options OK: ['patch', 'minor', 'major', 'alpha', 'beta', 'rc', 'stable']`

- [ ] **Step 3: Sanity-check uv bump semantics from the current prerelease**

The working tree is already `0.0.0a0` (committed in Task 1), so `--dry-run`
bumps read it without mutating anything. Run:

```bash
for b in alpha beta rc stable patch; do printf "%-7s -> " "$b"; uv version --short --dry-run --bump "$b"; done
```

Expected output:

```
alpha   -> 0.0.0a1
beta    -> 0.0.0b1
rc      -> 0.0.0rc1
stable  -> 0.0.0
patch   -> 0.0.1
```

(`--dry-run` does not write `pyproject.toml`; no restore needed.)

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/bump-version.yml
git commit -m "$(cat <<'EOF'
ci: add pre-release bump options to Bump Version workflow

Adds alpha/beta/rc/stable to the bump_type choices so alphas can be
iterated (0.0.0a0 -> 0.0.0a1) and promoted to stable without hand edits.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rewrite `publish.yml` (release trigger, target selection, version guard)

**Files:**
- Modify: `.github/workflows/publish.yml` (full rewrite of triggers, environment, and steps)

**Interfaces:**
- Consumes: `pyproject.toml` version via `tools/get-version.sh`; `github.event.release.tag_name` on release events; `inputs.target` on manual dispatch.
- Produces: a publish job that uploads to real PyPI on `release: published`, or to TestPyPI/PyPI on manual dispatch, gated by a tag/version guard on release events.

- [ ] **Step 1: Add a guard-logic shell test (TDD for the comparison)**

The version guard embeds a tag-strip-and-compare. Prove the logic in isolation first. Run:

```bash
check() {
  local VERSION="$1" TAG="$2"
  local TAG_VERSION="${TAG#v}"
  if [ "$VERSION" != "$TAG_VERSION" ]; then echo "MISMATCH"; return 1; else echo "MATCH"; return 0; fi
}
check 0.0.0a0 v0.0.0a0 && echo "case1 ok"      # expect MATCH then case1 ok
check 0.0.0a0 v0.0.0a1 || echo "case2 ok"      # expect MISMATCH then case2 ok
check 0.0.0a0 0.0.0a0  && echo "case3 ok"      # bare tag also MATCH (strip is a no-op)
```

Expected output:

```
MATCH
case1 ok
MISMATCH
case2 ok
MATCH
case3 ok
```

- [ ] **Step 2: Rewrite `publish.yml`**

Replace the entire contents of `.github/workflows/publish.yml` with:

```yaml
name: Publish

on:
  release:
    types: [published]
  workflow_dispatch:
    inputs:
      target:
        description: 'Publish target'
        required: true
        default: 'testpypi'
        type: choice
        options:
          - testpypi
          - pypi

permissions:
  id-token: write
  contents: read

jobs:
  publish:
    runs-on: ubuntu-latest
    # Release events always go to real PyPI. Manual dispatch honors `target`.
    # The environment name selects the matching Trusted-Publishing identity.
    environment:
      name: ${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi') && 'testpypi' || 'pypi' }}
      url: ${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi') && 'https://test.pypi.org/p/mononet' || 'https://pypi.org/p/mononet' }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v7

      - name: Set up Python
        uses: actions/setup-python@v6
        with:
          python-version: "3.13"

      - name: Install uv
        uses: astral-sh/setup-uv@v7

      - name: Get version from pyproject.toml
        id: version
        run: |
          VERSION=$(./tools/get-version.sh)
          echo "package_version=$VERSION" >> "$GITHUB_OUTPUT"
          echo "Package version: $VERSION"

      - name: Verify release tag matches version
        if: github.event_name == 'release'
        run: |
          VERSION="${{ steps.version.outputs.package_version }}"
          TAG="${{ github.event.release.tag_name }}"
          TAG_VERSION="${TAG#v}"
          echo "Release tag: $TAG (-> $TAG_VERSION); pyproject version: $VERSION"
          if [ "$VERSION" != "$TAG_VERSION" ]; then
            echo "::error::Release tag $TAG (-> $TAG_VERSION) does not match pyproject version $VERSION" >&2
            exit 1
          fi

      - name: Build package
        run: uv build

      - name: Verify build artifacts
        run: |
          ls -lh dist/
          test -f "dist/mononet-${{ steps.version.outputs.package_version }}-py3-none-any.whl"
          test -f "dist/mononet-${{ steps.version.outputs.package_version }}.tar.gz"

      - name: Publish
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          # Empty repository-url defaults to real PyPI; set to TestPyPI only on
          # a manual dispatch that selected the testpypi target.
          repository-url: ${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi') && 'https://test.pypi.org/legacy/' || '' }}
```

- [ ] **Step 3: Verify the YAML parses and the key fields are correct**

Run:

```bash
uv run python - <<'PY'
import yaml
d = yaml.safe_load(open(".github/workflows/publish.yml"))
on = d[True]  # `on` -> YAML 1.1 boolean True
assert on["release"]["types"] == ["published"], on["release"]
assert on["workflow_dispatch"]["inputs"]["target"]["options"] == ["testpypi","pypi"]
assert "push" not in on, "bare tag-push trigger must be removed"
steps = d["jobs"]["publish"]["steps"]
names = [s.get("name") for s in steps]
assert "Verify release tag matches version" in names, names
guard = next(s for s in steps if s.get("name") == "Verify release tag matches version")
assert guard["if"] == "github.event_name == 'release'", guard.get("if")
pub = next(s for s in steps if s.get("name") == "Publish")
assert "repository-url" in pub["with"], pub
print("publish.yml OK")
PY
```

Expected output: `publish.yml OK`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/publish.yml
git commit -m "$(cat <<'EOF'
ci: trigger publish on GitHub Release, add TestPyPI + version guard

Replaces the tag-push trigger with release:published plus a
target-selectable workflow_dispatch (testpypi default). Release events
always publish to real PyPI and fail unless the release tag matches the
pyproject version. Trusted Publishing identity is selected by environment.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add `docs/releasing.md` runbook; wire toctree; trim `CONTRIBUTING.md`

**Files:**
- Create: `docs/releasing.md`
- Modify: `docs/index.md:47-55` (toctree)
- Modify: `CONTRIBUTING.md:80-102` (release section → pointer)

**Interfaces:**
- Consumes: the finished `publish.yml`/`bump-version.yml` behavior from Tasks 2–3 (the runbook describes them).
- Produces: a Sphinx-rendered maintainer runbook reachable from the docs nav; `CONTRIBUTING.md` points to it.

- [ ] **Step 1: Create `docs/releasing.md`**

Create `docs/releasing.md` with exactly:

````markdown
# Releasing

Maintainer runbook for cutting a `mononet` release to PyPI. Publishing uses
**PyPI Trusted Publishing** (OIDC) — no API tokens or stored secrets.

## One-time setup (maintainer, web UI)

Do this once before the first release.

1. **GitHub Environments.** In the repository settings, create two
   environments: `pypi` and `testpypi`.
2. **PyPI pending publisher** — at <https://pypi.org/manage/account/publishing/>,
   add a pending publisher:
   - PyPI Project Name: `mononet`
   - Owner: `davorrunje`
   - Repository name: `mononet`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`
3. **TestPyPI pending publisher** — repeat at
   <https://test.pypi.org/manage/account/publishing/> with Environment name
   `testpypi`.

If any of these values is wrong, the first upload fails at the OIDC exchange
with a "not a trusted publisher" error.

## Version bumping

The version lives in `pyproject.toml` and is the single source of truth. Use
the **Bump Version** GitHub Action (Actions tab → Bump Version → Run workflow),
which runs `uv version --bump <type>` and opens a PR. Options:

| Bump | From `0.0.0a0` | Notes |
|------|----------------|-------|
| `alpha` | `0.0.0a1` | iterate the alpha |
| `beta` | `0.0.0b1` | |
| `rc` | `0.0.0rc1` | |
| `stable` | `0.0.0` | promote prerelease to final |
| `patch` | `0.0.1` | |
| `minor` | `0.1.0` | |
| `major` | `1.0.0` | |

`alpha`/`beta`/`rc`/`stable` only work standalone when the current version is
already a pre-release. From a stable version, bump a release component first
(`patch`/`minor`/`major`); a bare `alpha` bump from a stable version errors.

## Per-release flow

1. *(If benchmarks changed)* in a `gpu-*` devcontainer, run
   `./tools/execute-benchmarks.sh`, sanity-check `git diff docs/benchmarks/`,
   and commit the regenerated notebooks.
2. Run **Bump Version** with the desired bump type; merge the resulting PR into
   `main`.
3. *(Optional rehearsal)* Actions tab → **Publish** → Run workflow →
   `target: testpypi`. Verify the project page on
   <https://test.pypi.org/project/mononet/> and a clean install:
   `pip install -i https://test.pypi.org/simple/ mononet==<version>`.
4. Create a **GitHub Release**: tag `v<version>` (e.g. `v0.0.0a0`), target the
   merge commit on `main`, mark it **pre-release** for alphas/betas/rcs, and
   write release notes.
5. Publishing the release fires the **Publish** workflow to real PyPI. It fails
   fast if the tag does not match the `pyproject.toml` version. The **Docs**
   workflow deploys the versioned docs.

## Notes

- Publishing does not re-run the test matrix; releases are expected to be cut
  from a `main` whose `build.yml` checks are green.
- A real-PyPI upload cannot be replaced, only yanked. Use the TestPyPI
  rehearsal for first-of-its-kind releases.
````

- [ ] **Step 2: Add `releasing` to the docs toctree**

In `docs/index.md`, the toctree at lines 47–55 is:

```
```{toctree}
:hidden:

guides/index
concepts/index
benchmarks/index
reference
about/index
```
```

Insert `releasing` before `about/index` so it reads:

```
```{toctree}
:hidden:

guides/index
concepts/index
benchmarks/index
reference
releasing
about/index
```
```

- [ ] **Step 3: Replace the `CONTRIBUTING.md` release section with a pointer**

In `CONTRIBUTING.md`, replace the block spanning the current `## Release process`
heading through the end of the `### One-time PyPI setup (maintainer only)`
subsection (lines ~80–102, ending with the line
"After that, every tag-pushed release publishes via OIDC with no API tokens.")
with:

```markdown
## Release process

The full maintainer runbook — one-time Trusted-Publishing setup, the Bump
Version action, TestPyPI rehearsal, and the GitHub-Release-triggered publish —
lives in [`docs/releasing.md`](docs/releasing.md).
```

Leave the surrounding sections (the benchmarks note above, `## Commit messages`
below) intact.

- [ ] **Step 4: Build the docs strict and verify the page renders**

Run: `./tools/build-docs.sh`
Expected: build completes with no warnings-as-errors (exit 0).

Then verify the page was produced:

```bash
test -f docs/_build/html/releasing.html && echo "releasing page OK"
```

Expected output: `releasing page OK`

- [ ] **Step 5: Verify the CONTRIBUTING pointer**

Run: `grep -n "docs/releasing.md" CONTRIBUTING.md`
Expected: one match inside the `## Release process` section. Also confirm the
stale text is gone:

```bash
grep -c "tag-pushed release publishes via OIDC" CONTRIBUTING.md
```

Expected output: `0`

- [ ] **Step 6: Commit**

```bash
git add docs/releasing.md docs/index.md CONTRIBUTING.md
git commit -m "$(cat <<'EOF'
docs: add release runbook and reconcile CONTRIBUTING

New docs/releasing.md documents Trusted-Publishing setup, the Bump Version
action, TestPyPI rehearsal, and the GitHub-Release publish flow. Wired into
the Sphinx toctree; CONTRIBUTING release section now points to it.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the executor

- **No PyPI uploads happen during implementation.** This plan only changes
  files. The actual first upload (and the one-time pending-publisher
  registration) is a maintainer action performed after merge, per
  `docs/releasing.md`.
- If `./tools/build-docs.sh` flags the new page as an orphan or a broken
  cross-reference, fix the toctree wiring (Step 4.2) rather than excluding the
  page — the runbook must be reachable from the nav.
- `tools/get-version.sh` uses `grep -oP` (PCRE), which is unavailable on
  macOS/BSD grep but present on the CI Ubuntu runners. Local `./tools/get-version.sh`
  may fail on macOS; use `uv version --short` locally instead. This is
  pre-existing and out of scope.
