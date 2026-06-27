# PyPI Alpha Release Pipeline — Design

**Date:** 2026-06-27
**Status:** Approved
**Scope:** Ship the first alpha of `mononet` (`0.0.0a0`) to PyPI, and put the
release automation on a footing we keep for every subsequent release.

## 1. Goal

Make `mononet 0.0.0a0` installable from PyPI via `pip install mononet`, driven
by a GitHub Release, with a manual version-bump action that edits
`pyproject.toml` and a manual TestPyPI rehearsal path. Authentication is PyPI
Trusted Publishing (OIDC) — no long-lived secrets.

## 2. Background

Release infrastructure was scaffolded in commit `3122577` and is mostly in
place; this design reconciles it with the intended workflow rather than
building from scratch.

Already present:

- `.github/workflows/publish.yml` — triggers on tag push `v*.*.*` + manual
  dispatch; Trusted Publishing (`id-token: write`, environment `pypi`); builds
  with `uv build`; verifies artifact names; publishes via
  `pypa/gh-action-pypi-publish`.
- `.github/workflows/bump-version.yml` — manual dispatch, `patch`/`minor`/`major`,
  runs `uv version --bump`, opens a PR.
- `.github/workflows/build.yml` — full test matrix on PRs/`main`.
- `tools/get-version.sh` — extracts the version from `pyproject.toml`.
- `pyproject.toml` version `0.0.0`, `Development Status :: 3 - Alpha`.

Gaps this design closes: version is not `0.0.0a0`; publish triggers on tag-push
rather than the GitHub Release event; the bump action offers no pre-release
bumps; there is no TestPyPI rehearsal; the release runbook in `CONTRIBUTING.md`
describes the old tag-push flow.

## 3. Decisions

| Topic | Decision |
|---|---|
| Auth | Trusted Publishing (OIDC) on **both** PyPI and TestPyPI. No API tokens. |
| Publish trigger | `release: types: [published]` (real PyPI) + `workflow_dispatch` (target-selectable). Drop the bare `tags: v*.*.*` push trigger. |
| Version source of truth | `pyproject.toml`. Git tag must match it; the workflow enforces this on release events. |
| Tag convention | `v`-prefixed, e.g. `v0.0.0a0`. Guard strips the leading `v` before comparing. |
| Bump options | `major`, `minor`, `patch`, `alpha`, `beta`, `rc`, `stable` (all `uv version --bump` semantics). |
| TestPyPI | Manual `workflow_dispatch` with a `target` input (`testpypi` default, `pypi`). Release events always target real PyPI. |
| Test gating | Publish does **not** re-run the test matrix. Releases are cut from `main`, whose `build.yml` already passed via required checks. Documented assumption, not enforced in `publish.yml`. |
| Runbook location | Standalone `docs/releasing.md`, wired into the Sphinx toctree and pointed to from `CONTRIBUTING.md`. |

Non-goals: dynamic (git-tag-derived) versioning; automated CHANGELOG
generation; signing the published artifacts beyond what Trusted Publishing
already attests.

## 4. Changes

### 4.1 `pyproject.toml`

`version = "0.0.0"` → `version = "0.0.0a0"`. No classifier change
(`Development Status :: 3 - Alpha` already set). `0.0.0a0` is already a
PEP 440-normalized form, so wheel/sdist filenames become
`mononet-0.0.0a0-py3-none-any.whl` and `mononet-0.0.0a0.tar.gz` — matching the
existing artifact-verification step in `publish.yml`.

### 4.2 `.github/workflows/bump-version.yml`

Extend the `bump_type` choice list from `patch`/`minor`/`major` to:

```
major, minor, patch, alpha, beta, rc, stable
```

Default stays `patch`. The rest of the job is unchanged: `uv version --bump
<type>` then a `peter-evans/create-pull-request` PR titled
"Bump version to <new>". Reference flows from `0.0.0a0`:

- `--bump alpha` → `0.0.0a1`
- `--bump stable` → `0.0.0`
- `--bump patch` → `0.0.1`

### 4.3 `.github/workflows/publish.yml`

**Triggers:**

```yaml
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
        options: [testpypi, pypi]
```

The `push: tags: v*.*.*` trigger is removed.

**Target + environment resolution.** A `release` event always publishes to real
PyPI; a `workflow_dispatch` honors the `target` input. The job's `environment`
selects the matching Trusted-Publishing identity:

```yaml
environment:
  name: ${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi') && 'testpypi' || 'pypi' }}
  url: ${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi') && 'https://test.pypi.org/p/mononet' || 'https://pypi.org/p/mononet' }}
```

**Version guard (release events only).** Before building, compare the release
tag to the package version and fail on mismatch:

```bash
if [ "${{ github.event_name }}" = "release" ]; then
  VERSION=$(./tools/get-version.sh)
  TAG="${{ github.event.release.tag_name }}"
  TAG_VERSION="${TAG#v}"
  if [ "$VERSION" != "$TAG_VERSION" ]; then
    echo "::error::Release tag $TAG (-> $TAG_VERSION) does not match pyproject version $VERSION" >&2
    exit 1
  fi
fi
```

On `workflow_dispatch` this guard is skipped (no tag).

**Publish step.** Keep `uv build` and the artifact-name verification. The
publish step targets TestPyPI when selected:

```yaml
- name: Publish
  uses: pypa/gh-action-pypi-publish@release/v1
  with:
    repository-url: ${{ (github.event_name == 'workflow_dispatch' && inputs.target == 'testpypi') && 'https://test.pypi.org/legacy/' || '' }}
```

(An empty `repository-url` defaults to real PyPI.)

### 4.4 `docs/releasing.md` (new)

Maintainer runbook. Contents:

1. **One-time setup (manual, web UI):**
   - Create GitHub Environments `pypi` and `testpypi` in repo settings.
   - Register a **pending publisher** on PyPI (<https://pypi.org/manage/account/publishing/>):
     project `mononet`, owner `davorrunje`, repo `mononet`, workflow
     `publish.yml`, environment `pypi`.
   - Register the same on TestPyPI (<https://test.pypi.org/manage/account/publishing/>)
     with environment `testpypi`.
2. **Per-release flow:**
   1. (If benchmarks changed) re-execute notebooks in a `gpu-*` devcontainer
      via `./tools/execute-benchmarks.sh`, commit.
   2. Run **Bump Version** (choose bump type) → merge the resulting PR.
   3. *(Optional rehearsal)* dispatch **Publish** with `target: testpypi`;
      verify the page and a clean install from TestPyPI.
   4. Create a **GitHub Release** with tag `v<version>` (e.g. `v0.0.0a0`),
      mark pre-release for alphas, write release notes.
   5. Publish fires → real PyPI. The `Docs` workflow deploys versioned docs.

### 4.5 `docs/index.md` toctree

Add `releasing` to the documentation toctree so the runbook renders in the
Sphinx site.

### 4.6 `CONTRIBUTING.md`

Replace the existing `## Release process` and `### One-time PyPI setup` block
(lines ~80–102, which describe the superseded tag-push flow and omit TestPyPI)
with a short pointer to `docs/releasing.md`.

## 5. Acceptance

- `pyproject.toml` reports `0.0.0a0`; `uv build` produces
  `mononet-0.0.0a0-*` artifacts.
- Bump Version workflow lists all seven bump types and opens a PR.
- Publish workflow: a TestPyPI manual dispatch uploads to TestPyPI; a GitHub
  Release tagged `v0.0.0a0` uploads to real PyPI; a release whose tag mismatches
  the pyproject version fails the version-guard step.
- `docs/releasing.md` builds under `./tools/build-docs.sh` (strict) and is
  reachable from the docs nav; `CONTRIBUTING.md` points to it.
- `mononet` is installable: `pip install mononet==0.0.0a0`.

## 6. Risks

- **First upload is irreversible.** A bad real-PyPI upload can only be yanked,
  not replaced. The TestPyPI rehearsal step is the mitigation.
- **Pending-publisher misconfiguration** (wrong environment/workflow name)
  surfaces only at first upload as an OIDC rejection. The runbook lists exact
  values to register.
