# Deploy Documentation to GitHub Pages — Design

**Status:** Draft for review
**Date:** 2026-05-22
**Owner:** Davor Runje
**Triggering event:** Post-MkDocs-to-Sphinx migration ([sphinx-migration-design](2026-05-22-sphinx-migration-design.md)), the `Docs` workflow fails on every push (`sphinx-multiversion` crashes on current Sphinx with `TypeError: Config.read() takes 2 positional arguments but 3 were given`) and GitHub Pages is not yet enabled for the repository — so even on green builds nothing serves.

## Goal

Publish `mononet` documentation at `https://davorrunje.github.io/mononet/` with:

- A multi-version layout (`/main/` for the development branch, `/v<X.Y.Z>/` per released tag).
- A working PyData theme version switcher driven by `versions.json`.
- A green, repeatable CI pipeline that rebuilds and redeploys on push to `main` and on any `v*.*.*` tag.

## Non-goals

- Custom domain / DNS changes.
- Search backend changes (Sphinx's built-in search stays).
- Documentation content changes.
- Backfilling builds for historical tags that pre-date the Sphinx migration. Only future `v*.*.*` tags are in scope.
- Preserving the existing `gh-pages` branch contents. The branch will be obsoleted by this change.

## Background

Current state (2026-05-22):

- `.github/workflows/docs.yml` builds with `sphinx-multiversion`, calls `tools/gen_versions_json.py`, writes a root `index.html` redirect, and pushes the build to a `gh-pages` branch via `peaceiris/actions-gh-pages@v4`.
- The build step crashes because `sphinx-multiversion` (last release 2022) is incompatible with the Sphinx version pulled in by `uv sync --group docs`. The project is unmaintained.
- `gh api repos/davorrunje/mononet/pages` returns `404` — GitHub Pages has never been enabled for this repository. The pre-existing `origin/gh-pages` branch contains only stale MkDocs/mike deploys from before the Sphinx migration.
- `docs/conf.py` already declares the multi-version intent: `smv_branch_whitelist = r"^main$"`, `smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"`, `smv_outputdir_format = "{ref.name}"`, plus a PyData switcher pointing at `https://davorrunje.github.io/mononet/versions.json`.
- No `v*.*.*` tags exist on the repository yet, so today multi-version builds in practice produce a single `main/` directory.

## Design overview

Two layered changes:

1. **Replace `sphinx-multiversion` with `sphinx-polyversion`** — its maintained successor. `sphinx-polyversion` uses a Python driver file (`docs/poly.py`) instead of `smv_*` config keys, so the multi-version configuration moves out of `conf.py` and into a new driver.
2. **Replace the `gh-pages` branch deploy with the official GitHub Pages Actions pipeline** — `actions/configure-pages` → `actions/upload-pages-artifact` → `actions/deploy-pages`. No `gh-pages` branch is written; Pages serves the per-run artifact directly.

## Workflow topology

Single workflow `.github/workflows/docs.yml`, two jobs.

```
job: build
  runs-on: ubuntu-latest
  permissions:
    contents: read
  steps:
    - actions/checkout@v6 (fetch-depth: 0, fetch-tags: true)
    - actions/setup-python@v6 (python-version: "3.13")
    - astral-sh/setup-uv@v7
    - uv sync --group docs --extra all
    - uv run sphinx-polyversion docs/poly.py
        env: DOCS_VERSION computed per-build by the driver
    - uv run python tools/gen_versions_json.py docs/_build/html https://davorrunje.github.io/mononet
    - write docs/_build/html/index.html (meta-refresh redirect → ./main/)
    - actions/configure-pages@v5 (enablement: true)
    - actions/upload-pages-artifact@v3 (path: docs/_build/html)

job: deploy
  needs: build
  runs-on: ubuntu-latest
  environment:
    name: github-pages
    url: ${{ steps.deployment.outputs.page_url }}
  permissions:
    pages: write
    id-token: write
  steps:
    - id: deployment
      uses: actions/deploy-pages@v4
```

Triggers stay the same: `push` to `main`, `push` of `v*.*.*` tags, and `workflow_dispatch`. The `concurrency` group `docs-${{ github.ref }}` with `cancel-in-progress: true` is preserved.

Permission shape changes from the current `contents: write` (needed to push to `gh-pages`) to `contents: read` on `build` plus `pages: write` + `id-token: write` on `deploy` — least-privilege per the Pages Actions model.

## sphinx-polyversion driver (`docs/poly.py`)

New file at `docs/poly.py` replaces the `# -- sphinx-multiversion` block in `conf.py`. Responsibilities:

- Declare the source directory (`docs/`) and the output directory (`docs/_build/html/`).
- Whitelist refs to build:
  - branches matching `^main$`
  - remote `origin` only
  - tags matching `^v\d+\.\d+\.\d+$`
- Per-ref output directory: `{ref.name}/` so the layout becomes `docs/_build/html/{main, v1.2.3, ...}/` — preserving what `tools/gen_versions_json.py` already scans.
- Each per-version `sphinx-build` invocation receives the env var `DOCS_VERSION` set to the ref name, so the existing switcher logic in `docs/conf.py` (`os.environ.get("DOCS_VERSION", "latest")`) keeps working without modification.
- `sphinx-build` runs with `-W` (warnings-as-errors), matching `tools/build-docs.sh`.

## `conf.py` changes

- Delete the `# -- sphinx-multiversion -------------------------------------------------` block (the `smv_*` settings on lines 103–109).
- No other changes. The `DOCS_VERSION` env hook and the switcher `json_url` stay as-is.

## Dependency changes (`pyproject.toml`)

- Remove `sphinx-multiversion>=0.2` from the `docs` dependency group.
- Add `sphinx-polyversion` (current stable; pin floor only).
- Regenerate `uv.lock` via `uv sync --group docs`.

## `tools/` changes

- `tools/gen_versions_json.py`: no logic change required — it already scans the build directory for subdirectories matching `^v\d+\.\d+\.\d+$` plus a `main/` entry, which is exactly the layout `sphinx-polyversion` will produce.
- `tools/build-docs.sh`: unchanged. Local single-version preview still uses plain `sphinx-build`.
- `tools/serve-docs.sh`: unchanged.

## Pages enablement & `gh-pages` branch fate

- `actions/configure-pages@v5` is invoked with `enablement: true`. On the first successful run, this creates the Pages site configured for "GitHub Actions" as the source. No manual repo-settings step is required, assuming the workflow's `GITHUB_TOKEN` retains default site-creation rights.
- If org policy blocks programmatic enablement, the fallback is a one-time manual enablement in repo Settings → Pages → Source: "GitHub Actions". This is documented as a rollback note, not part of the routine flow.
- The pre-existing `origin/gh-pages` branch is **not** written to anymore. After the first successful Actions-based deploy verifies the new pipeline serves correctly, the branch is deleted (`git push origin :gh-pages`). The deletion is a one-time manual step performed by the owner, **not** automated by this workflow — destroying historical branches from inside CI is a blast-radius concern not worth automating.

## Root index redirect

The repository owner expects `https://davorrunje.github.io/mononet/` (no trailing version) to land users on `/main/` until tagged releases exist. `sphinx-polyversion` may emit its own root index; the workflow's redirect-writing step runs **after** the polyversion build and overwrites whatever lands at `docs/_build/html/index.html` with the meta-refresh redirect, identical to the current workflow's step. Once tagged releases exist, this redirect can switch to point at the newest tag — out of scope for this change.

## Testing & rollout

**Local validation (pre-merge):**

- `uv sync --group docs --extra all`
- `uv run sphinx-polyversion docs/poly.py` produces `docs/_build/html/main/index.html` (and per-tag dirs if tags exist).
- `uv run python tools/gen_versions_json.py docs/_build/html https://davorrunje.github.io/mononet` produces `docs/_build/html/versions.json` with a `dev (main)` entry.
- Open `docs/_build/html/main/index.html` in a browser; confirm the PyData version switcher renders and (with a local server) loads `versions.json`.

**CI validation:**

1. Merge the change to `main`. Workflow fires automatically.
2. If the build job succeeds but Pages enablement fails (org policy), enable Pages manually once and re-run `workflow_dispatch`.
3. Visit `https://davorrunje.github.io/mononet/` and confirm the redirect to `/main/` works, the page renders, and the switcher dropdown is populated.

**Rollback:**

- If `sphinx-polyversion` regresses, the documented escape hatch is to replace the polyversion invocation with a plain `uv run sphinx-build -W docs docs/_build/html/main` and skip the per-tag loop. Single-version deploy, switcher disabled, but docs still serve.

## Risks

- **`sphinx-polyversion` config drift.** Its config model is not a drop-in for `smv_*`. A working `docs/poly.py` has to be authored and verified locally before the workflow can rely on it.
- **First-time Pages enablement permissions.** If `actions/configure-pages` cannot create the site under default `GITHUB_TOKEN` scope (e.g. org-level restriction), the first deploy fails until the owner enables Pages manually. Mitigation: documented above; only affects the very first deploy.
- **Polyversion root output collision.** Polyversion may write its own `index.html` at the build root. The redirect-writing step deliberately runs last to overwrite it; if polyversion's output structure differs from expectation, the step still wins because it's a plain file write.
- **No tags yet.** Multi-version infrastructure goes in before any `v*.*.*` tag exists, so the "switcher" will look thin until the first release. This is the right ordering — having the pipeline ready means the first tag deploys without additional plumbing.

## Acceptance criteria

- [ ] `Docs` workflow runs green on push to `main`.
- [ ] `https://davorrunje.github.io/mononet/` returns HTTP 200 and redirects to `/main/`.
- [ ] `https://davorrunje.github.io/mononet/main/` renders the Sphinx site with the PyData theme.
- [ ] `https://davorrunje.github.io/mononet/versions.json` exists and validates (one `dev (main)` entry minimum).
- [ ] The PyData version switcher dropdown is populated in the rendered site.
- [ ] No commits are pushed to `gh-pages` by the new workflow.
- [ ] Old `origin/gh-pages` branch is deleted (manual one-time step, performed after acceptance).
