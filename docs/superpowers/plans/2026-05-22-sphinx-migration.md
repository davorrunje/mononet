# Sphinx Documentation Migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MkDocs + `mkdocs-material` documentation stack with Sphinx + PyData Sphinx Theme + `sphinx-autodoc2` + `myst-nb` + `sphinx-multiversion`, in-place, with every commit leaving the pre-commit `docs` hook green.

**Architecture:** Two-stage swap. Tasks 1–4 bring up Sphinx **alongside** MkDocs (Sphinx reads from `docs/docs/` source; both engines buildable, MkDocs still authoritative for pre-commit). Task 5 swaps `tools/build-docs.sh` over to Sphinx (pre-commit now uses Sphinx). Tasks 6–7 remove MkDocs and move content up to `docs/`. Tasks 8–13 polish, version, deploy, document.

**Tech Stack:** Sphinx, PyData Sphinx Theme, `sphinx-autodoc2`, `myst-nb`, `sphinx-design`, `sphinx-copybutton`, `sphinx-togglebutton`, `sphinx-multiversion`, `linkify-it-py`. Python 3.11+, `uv` for dependency management.

**Spec:** [docs/superpowers/specs/2026-05-22-sphinx-migration-design.md](../specs/2026-05-22-sphinx-migration-design.md)

---

## File map

**Created:**
- `docs/conf.py` — Sphinx configuration (replaces `docs/mkdocs.yml` + `docs/docs.py`)
- `docs/_static/extra.css` — moved from `docs/docs/stylesheets/extra.css`
- `docs/_static/extra.js` — moved from `docs/docs/javascripts/extra.js`
- `docs/_static/logo.png` — moved from `docs/docs/assets/img/logo.png`
- `docs/_templates/layout.html` — OG/Twitter meta tag overrides (replaces `docs/overrides/main.html`)
- `docs/_templates/404.html` — custom 404 page (replaces `docs/overrides/404.html`)
- `docs/index.md` — top-level toctree entry (replaces `docs/docs/SUMMARY.md`)
- `tools/gen_versions_json.py` — post-build helper that writes `versions.json` for the PyData theme version switcher

**Modified:**
- `pyproject.toml` — `docs` dependency group rewritten
- `tools/build-docs.sh` — calls `sphinx-build` / `sphinx-multiversion` instead of `python docs.py build`
- `tools/serve-docs.sh` — calls `sphinx-autobuild`
- `.github/workflows/docs.yml` — uses Sphinx + multiversion + custom deploy step
- `CLAUDE.md` — docs commands section
- `.gitignore` — add `docs/apidocs/`, remove obsolete entries

**Deleted (after Phase 3):**
- `docs/mkdocs.yml`
- `docs/docs.py`
- `docs/create_api_docs.py`
- `docs/__init__.py`
- `docs/overrides/` (entire directory)
- `docs/docs/navigation_template.txt`
- `docs/docs/SUMMARY.md` (already gitignored; remove from `.gitignore`)
- `docs/docs/api/` directory (already gitignored)
- `docs/docs/stylesheets/`, `docs/docs/javascripts/`, `docs/docs/assets/` (after copy to `_static/`)
- `docs/site/` directory (build output, already gitignored)
- `docs/docs/` directory (after content moves up one level)

---

## Phase 1 — Bring up Sphinx alongside MkDocs

The whole phase preserves the working MkDocs build. Each commit's pre-commit `docs` hook keeps invoking `tools/build-docs.sh` → `docs.py build` → `mkdocs build`, which works because we don't touch any MkDocs file.

---

### Task 1: Add Sphinx dependencies to the `docs` group

**Files:**
- Modify: `pyproject.toml:63-75`

- [ ] **Step 1: Replace the `docs` dependency group with both stacks present**

  Edit `pyproject.toml`, replace lines 63–75 (the current `docs = [...]` block) with:

  ```toml
  docs = [
      # MkDocs stack (kept temporarily — removed in Task 6)
      "mkdocs-material==9.7.6",
      "mkdocstrings==1.0.4",
      "mkdocstrings-python==2.0.3",
      "mkdocs-literate-nav==0.6.3",
      "mkdocs-glightbox==0.5.2",
      "mkdocs-jupyter>=0.25",
      "mdx-include==1.4.2",
      "mkdocs-git-revision-date-localized-plugin==1.5.2",
      "typer==0.25.1",
      "mkdocs-minify-plugin==0.8.0",
      "mike==2.2.0",
      # Sphinx stack
      "sphinx>=8.1",
      "pydata-sphinx-theme>=0.16",
      "sphinx-autodoc2>=0.5",
      "myst-nb>=1.1",
      "sphinx-copybutton>=0.5",
      "sphinx-design>=0.6",
      "sphinx-togglebutton>=0.3",
      "sphinx-autobuild>=2024.10",
      "sphinx-multiversion>=0.2",
      "linkify-it-py>=2.0",
  ]
  ```

- [ ] **Step 2: Sync dependencies**

  Run: `uv sync --group docs`
  Expected: completes successfully; both stacks installed in `.venv`.

- [ ] **Step 3: Verify both engines are importable**

  Run:
  ```bash
  uv run python -c "import mkdocs; import sphinx; import autodoc2; import myst_nb; import pydata_sphinx_theme; print('ok')"
  ```
  Expected: prints `ok`.

- [ ] **Step 4: Verify the old MkDocs build still passes**

  Run: `./tools/build-docs.sh`
  Expected: MkDocs build completes (the ProperDocs warning is still printed; that's fine).

- [ ] **Step 5: Commit**

  ```bash
  git add pyproject.toml uv.lock
  git commit -m "build(docs): add Sphinx dependencies alongside MkDocs"
  ```

---

### Task 2: Create the Sphinx config skeleton

**Files:**
- Create: `docs/conf.py`
- Create: `docs/_static/.gitkeep`
- Create: `docs/_templates/.gitkeep`
- Modify: `.gitignore`

- [ ] **Step 1: Create `docs/conf.py` with minimal config pointing at the existing source tree**

  Write the following to `docs/conf.py`:

  ```python
  """Sphinx configuration for mononet documentation."""

  from __future__ import annotations

  import os
  from pathlib import Path

  # -- Project information ---------------------------------------------------
  project = "mononet"
  author = "Davor Runje"
  copyright = "2026, Davor Runje"
  html_baseurl = "https://davorrunje.github.io/mononet/"

  # -- Source layout ---------------------------------------------------------
  # Temporary: source files still live in docs/docs/ (existing MkDocs layout).
  # After Task 7, content moves up one level and this file becomes the source
  # directory itself.
  _HERE = Path(__file__).resolve().parent
  _SOURCE_SUFFIX_DIR = _HERE / "docs"
  if _SOURCE_SUFFIX_DIR.is_dir():
      # Phase 1–2: point at docs/docs/
      master_doc = "index"

  # -- General configuration -------------------------------------------------
  extensions = [
      "myst_nb",
      "autodoc2",
      "sphinx.ext.intersphinx",
      "sphinx.ext.mathjax",
      "sphinx_copybutton",
      "sphinx_design",
      "sphinx_togglebutton",
  ]
  exclude_patterns = ["_build", "site", "navigation_template.txt", "SUMMARY.md"]
  templates_path = ["_templates"]
  source_suffix = {
      ".md": "markdown",
      ".ipynb": "myst-nb",
  }

  # -- HTML output -----------------------------------------------------------
  html_theme = "pydata_sphinx_theme"
  html_static_path = ["_static"]
  html_title = "mononet"
  html_theme_options = {
      "logo": {
          "image_light": "_static/logo.png",
          "image_dark": "_static/logo.png",
      },
      "github_url": "https://github.com/davorrunje/mononet",
      "use_edit_page_button": True,
      "navbar_align": "left",
      "show_version_warning_banner": True,
      "switcher": {
          "json_url": "https://davorrunje.github.io/mononet/versions.json",
          "version_match": os.environ.get("DOCS_VERSION", "latest"),
      },
      "show_toc_level": 2,
      "pygments_light_style": "default",
      "pygments_dark_style": "monokai",
  }
  html_context = {
      "github_user": "davorrunje",
      "github_repo": "mononet",
      "github_version": "main",
      "doc_path": "docs",
  }
  html_css_files = ["extra.css"]
  html_js_files = ["extra.js"]

  # -- MyST configuration ----------------------------------------------------
  myst_enable_extensions = [
      "colon_fence",
      "deflist",
      "tasklist",
      "fieldlist",
      "dollarmath",
      "amsmath",
      "linkify",
      "substitution",
  ]
  myst_heading_anchors = 3

  # -- MyST-NB (notebooks) ---------------------------------------------------
  nb_execution_mode = "off"

  # -- sphinx-autodoc2 -------------------------------------------------------
  autodoc2_packages = [
      {"path": "../mononet", "auto_mode": True},
  ]
  autodoc2_render_plugin = "myst"
  autodoc2_docstring_parser_regexes = [
      (r".*", "google"),
  ]
  autodoc2_hidden_objects = ["private", "dunder"]
  autodoc2_index_template = None  # let Sphinx handle the index via toctree

  # -- intersphinx -----------------------------------------------------------
  intersphinx_mapping = {
      "python": ("https://docs.python.org/3", None),
      "torch": ("https://pytorch.org/docs/stable", None),
      "jax": ("https://docs.jax.dev", None),
      "keras": ("https://keras.io/api", None),
      "numpy": ("https://numpy.org/doc/stable", None),
  }
  intersphinx_disabled_reftypes = ["std:doc"]

  # -- sphinx-multiversion ---------------------------------------------------
  smv_branch_whitelist = r"^main$"
  smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"
  smv_released_pattern = r"^refs/tags/v\d+\.\d+\.\d+$"
  smv_remote_whitelist = r"^origin$"
  smv_outputdir_format = "{ref.name}"
  ```

- [ ] **Step 2: Create placeholder files so empty directories are tracked**

  ```bash
  touch docs/_static/.gitkeep docs/_templates/.gitkeep
  ```

- [ ] **Step 3: Update `.gitignore`**

  Edit `.gitignore`. Find the existing `docs/_build/` entry — leave it. Add these new lines near it:

  ```
  docs/apidocs/
  ```

  Leave `docs/docs/SUMMARY.md` and `docs/docs/api/` entries alone for now (still generated by MkDocs).

- [ ] **Step 4: Verify MkDocs still builds (sanity check, no Sphinx run yet)**

  Run: `./tools/build-docs.sh`
  Expected: passes as before. New `docs/conf.py` is ignored by MkDocs.

- [ ] **Step 5: Commit**

  ```bash
  git add docs/conf.py docs/_static/.gitkeep docs/_templates/.gitkeep .gitignore
  git commit -m "build(docs): add Sphinx config skeleton"
  ```

---

### Task 3: Copy static assets to `_static/`

We copy rather than move so MkDocs keeps working. Originals are deleted in Task 6.

**Files:**
- Create: `docs/_static/extra.css` (copy of `docs/docs/stylesheets/extra.css`)
- Create: `docs/_static/extra.js` (copy of `docs/docs/javascripts/extra.js`)
- Create: `docs/_static/logo.png` (copy of `docs/docs/assets/img/logo.png`)

- [ ] **Step 1: Copy the files**

  ```bash
  cp docs/docs/stylesheets/extra.css docs/_static/extra.css
  cp docs/docs/javascripts/extra.js docs/_static/extra.js
  cp docs/docs/assets/img/logo.png docs/_static/logo.png
  ```

- [ ] **Step 2: Remove the `.gitkeep` from `_static/` (now has real files)**

  ```bash
  rm docs/_static/.gitkeep
  ```

- [ ] **Step 3: Verify MkDocs build still works**

  Run: `./tools/build-docs.sh`
  Expected: still passes.

- [ ] **Step 4: Commit**

  ```bash
  git add docs/_static/
  git commit -m "build(docs): copy static assets to docs/_static for Sphinx"
  ```

---

### Task 4: Get the first successful Sphinx build

Create a temporary `index_sphinx.md` and `conf.py` source directory so Sphinx reads from `docs/docs/` (where content currently lives) but uses our new `docs/conf.py`. Iterate until `-W` clean.

**Files:**
- Create: `docs/docs/index_sphinx.md` (temporary; replaced in Task 7)
- Modify: `docs/conf.py` (point source directory at `docs/docs/`)

- [ ] **Step 1: Update `docs/conf.py` to use `docs/docs/` as the source directory**

  Sphinx's source directory is whatever directory contains `conf.py` by default. We want Sphinx to read content from `docs/docs/` but keep its config in `docs/`. The cleanest way is to invoke Sphinx with explicit source and config dirs:

  - **source dir**: `docs/docs/`
  - **config dir**: `docs/` (via `-c docs`)
  - **output dir**: `docs/_build/html/`

  No `conf.py` change is required for this — the split is at the command line. But add this near the top of `conf.py` to document the temporary state:

  ```python
  # NOTE: Phase 1–2 invocation:
  #   sphinx-build -c docs docs/docs docs/_build/html
  # After Task 7 (content moves up), invocation becomes:
  #   sphinx-build docs docs/_build/html
  ```

  Insert this comment block right under `"""Sphinx configuration for mononet documentation."""`.

- [ ] **Step 2: Create a temporary Sphinx landing page**

  Write the following to `docs/docs/index_sphinx.md`:

  ````markdown
  ---
  hide-toc: false
  ---

  # mononet

  **Unconstrained monotonic neural networks** with first-class support for
  **PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

  Reference implementation of:

  > Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
  > Neural Networks.* ICML 2023. [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

  ## Install

  ```
  pip install "mononet[torch]"      # PyTorch
  pip install "mononet[jax]"        # JAX + Flax NNX
  pip install "mononet[keras]"      # Keras 3
  pip install "mononet[all]"        # all three
  ```

  ```{toctree}
  :maxdepth: 1
  :caption: Getting started

  guides/pytorch
  guides/jax
  guides/keras
  ```

  ```{toctree}
  :maxdepth: 1
  :caption: Concepts

  concepts/monotonicity
  concepts/layers
  ```

  ```{toctree}
  :maxdepth: 1
  :caption: Benchmarks

  benchmarks/index
  ```

  ```{toctree}
  :maxdepth: 2
  :caption: API reference

  apidocs/index
  ```

  ```{toctree}
  :maxdepth: 1
  :caption: About

  about/license
  about/changelog
  about/citation
  contributing
  ```
  ````

  Note: the `apidocs/index` entry is created by `sphinx-autodoc2` at build time.

- [ ] **Step 3: Update `conf.py` to use the temporary index**

  In `docs/conf.py`, find the line `master_doc = "index"` (inside the `if _SOURCE_SUFFIX_DIR.is_dir():` block). Replace it with:

  ```python
      master_doc = "index_sphinx"
  ```

  Also, in the `exclude_patterns` list, add the original `index.md` so Sphinx ignores it during Phase 1:

  ```python
  exclude_patterns = [
      "_build", "site",
      "navigation_template.txt", "SUMMARY.md",
      "index.md",  # MkDocs landing page; Phase 1 uses index_sphinx.md
  ]
  ```

- [ ] **Step 4: First Sphinx build (no warnings-as-errors yet — triage warnings first)**

  Run: `uv run sphinx-build -c docs docs/docs docs/_build/html`
  Expected: build runs; may emit some warnings (broken cross-refs in existing markdown that refers to `../api/...`).

- [ ] **Step 5: Inspect the build output**

  Run: `ls docs/_build/html/`
  Expected output includes: `index_sphinx.html`, `apidocs/`, `guides/`, `concepts/`, `benchmarks/`, `about/`, `contributing.html`, `_static/`.

  Confirm the API docs were generated:
  Run: `ls docs/_build/html/apidocs/ | head`
  Expected: HTML files corresponding to `mononet.*` modules.

- [ ] **Step 6: Triage and fix warnings until `-W` clean**

  Re-run with warnings-as-errors:
  Run: `uv run sphinx-build -c docs -W docs/docs docs/_build/html`

  Common warnings and fixes:
  - **"toctree contains reference to nonexisting document"** — a referenced doc is in `exclude_patterns` or genuinely missing; fix the toctree entry.
  - **"document isn't included in any toctree"** — add it to the toctree in `index_sphinx.md`, or add to `exclude_patterns`.
  - **"undefined label" or "reference target not found"** for `../api/...` links inside existing `.md` files — these are fixed in Task 8. For now, add `suppress_warnings = ["myst.xref_missing"]` to `conf.py` temporarily; remove it in Task 8.
  - **"could not import module"** in `autodoc2` — verify `mononet` is importable; run `uv run python -c "import mononet"`. If it fails, the package needs to be installed in dev mode (it likely already is via `uv sync`).

  Add to `conf.py` if cross-ref warnings dominate:
  ```python
  suppress_warnings = ["myst.xref_missing"]  # TEMPORARY: removed in Task 8
  ```

  Iterate until `sphinx-build -c docs -W docs/docs docs/_build/html` exits 0.

- [ ] **Step 7: Verify MkDocs build still works**

  Run: `./tools/build-docs.sh`
  Expected: MkDocs build still passes (it ignores `index_sphinx.md` since it's not in MkDocs nav — though MkDocs may emit a "not in nav" info message; this is fine).

  If MkDocs errors on `index_sphinx.md`, add it to MkDocs' exclude list. In `docs/mkdocs.yml`, find the `exclude_docs:` block (lines 18–20) and add `index_sphinx.md`:

  ```yaml
  exclude_docs: |
    navigation_template.txt
    SUMMARY.md
    index_sphinx.md
  ```

- [ ] **Step 8: Open the Sphinx build in a browser to spot-check**

  Run: `uv run python -m http.server -d docs/_build/html 8009`
  Open: `http://localhost:8009/index_sphinx.html`
  Manually verify:
  - Landing page renders.
  - Light/dark mode toggle present.
  - Sidebar shows the toctree sections.
  - At least one API page (under `apidocs/`) renders.
  - Notebook page (`benchmarks/00-overview.html`) renders with outputs.

  Stop the server with Ctrl-C.

- [ ] **Step 9: Commit**

  ```bash
  git add docs/conf.py docs/docs/index_sphinx.md docs/mkdocs.yml
  git commit -m "build(docs): first successful Sphinx build alongside MkDocs"
  ```

---

## Phase 2 — Swap pre-commit to Sphinx

After this phase, the pre-commit `docs` hook invokes Sphinx. MkDocs files still exist but are unused; removed in Phase 3.

---

### Task 5: Swap build/serve scripts to Sphinx

**Files:**
- Modify: `tools/build-docs.sh`
- Modify: `tools/serve-docs.sh`

- [ ] **Step 1: Rewrite `tools/build-docs.sh`**

  Replace the entire content of `tools/build-docs.sh` with:

  ```bash
  #!/usr/bin/env bash
  set -e
  set -x

  # During Phase 2 the source dir is still docs/docs (Task 7 moves content up).
  if [[ -d docs/docs ]]; then
    uv run sphinx-build -c docs -W docs/docs docs/_build/html
  else
    uv run sphinx-build -W docs docs/_build/html
  fi
  ```

- [ ] **Step 2: Rewrite `tools/serve-docs.sh`**

  Replace the entire content of `tools/serve-docs.sh` with:

  ```bash
  #!/usr/bin/env bash
  set -e
  set -x

  if [[ -d docs/docs ]]; then
    uv run sphinx-autobuild -c docs docs/docs docs/_build/html \
      --port 8008 --host 0.0.0.0 \
      --watch mononet
  else
    uv run sphinx-autobuild docs docs/_build/html \
      --port 8008 --host 0.0.0.0 \
      --watch mononet
  fi
  ```

- [ ] **Step 3: Make sure scripts are executable**

  Run: `chmod +x tools/build-docs.sh tools/serve-docs.sh`

- [ ] **Step 4: Verify the new build script works end-to-end**

  Run: `./tools/build-docs.sh`
  Expected: Sphinx build passes with `-W` (warnings-as-errors). Output: `Documentation built in N seconds`.

  Confirm no MkDocs warning appears in the output (since `mkdocs` is no longer invoked).

- [ ] **Step 5: Verify the serve script starts (kill after 5 seconds)**

  Run: `timeout 5 ./tools/serve-docs.sh || true`
  Expected: prints `Serving on http://0.0.0.0:8008` (or similar) before the timeout kills it.

- [ ] **Step 6: Commit**

  ```bash
  git add tools/build-docs.sh tools/serve-docs.sh
  git commit -m "build(docs): swap build/serve scripts to Sphinx"
  ```

  Pre-commit's `docs` hook now invokes Sphinx via the new `build-docs.sh`. The MkDocs build pipeline is no longer exercised.

---

## Phase 3 — Remove MkDocs

Each commit keeps Sphinx building. We delete in two passes — first the orphaned Python (Task 6), then the content move (Task 7).

---

### Task 6: Remove MkDocs dependencies and orchestration files

**Files:**
- Modify: `pyproject.toml:63-90` (rewrite docs group)
- Delete: `docs/mkdocs.yml`
- Delete: `docs/docs.py`
- Delete: `docs/create_api_docs.py`
- Delete: `docs/__init__.py`
- Delete: `docs/overrides/` (entire directory)
- Delete: `docs/docs/navigation_template.txt`
- Modify: `.gitignore` (remove obsolete MkDocs entries)

- [ ] **Step 1: Rewrite the `docs` dependency group**

  Edit `pyproject.toml`. Replace the current `docs = [...]` block (still containing the MkDocs + Sphinx mix from Task 1) with:

  ```toml
  docs = [
      "sphinx>=8.1",
      "pydata-sphinx-theme>=0.16",
      "sphinx-autodoc2>=0.5",
      "myst-nb>=1.1",
      "sphinx-copybutton>=0.5",
      "sphinx-design>=0.6",
      "sphinx-togglebutton>=0.3",
      "sphinx-autobuild>=2024.10",
      "sphinx-multiversion>=0.2",
      "linkify-it-py>=2.0",
  ]
  ```

- [ ] **Step 2: Sync dependencies (removes MkDocs from the venv)**

  Run: `uv sync --group docs`
  Expected: removes `mkdocs`, `mkdocs-material`, `mkdocstrings`, `mike`, etc. from `.venv`.

- [ ] **Step 3: Delete MkDocs orchestration files**

  ```bash
  rm docs/mkdocs.yml
  rm docs/docs.py
  rm docs/create_api_docs.py
  rm docs/__init__.py
  rm -r docs/overrides/
  rm docs/docs/navigation_template.txt
  ```

- [ ] **Step 4: Clean up `.gitignore`**

  Edit `.gitignore`. Remove these now-obsolete lines:
  ```
  docs/docs/SUMMARY.md
  docs/docs/api/
  docs/site/
  /site
  ```

  Keep:
  ```
  docs/_build/
  docs/apidocs/
  ```

  Also remove any tracked but now-obsolete `docs/site/` directory:
  ```bash
  rm -rf docs/site/
  ```

- [ ] **Step 5: Delete the now-obsolete check-yaml exclusion in pre-commit**

  Edit `.pre-commit-config.yaml`. Find the line `exclude: 'docs/mkdocs.yml'` (under the `check-yaml` hook) and remove it. The line above (`-   id: check-yaml`) and below remain.

- [ ] **Step 6: Verify the Sphinx build still passes**

  Run: `./tools/build-docs.sh`
  Expected: passes with `-W` clean.

- [ ] **Step 7: Verify the linter / static analysis still pass**

  Run: `uv run ruff check --exit-non-zero-on-fix`
  Expected: passes (no orphan imports of deleted modules).

  Run: `uv run mypy`
  Expected: passes.

- [ ] **Step 8: Commit (stages additions, modifications, and deletions)**

  ```bash
  git add -A pyproject.toml uv.lock .gitignore .pre-commit-config.yaml \
              docs/mkdocs.yml docs/docs.py docs/create_api_docs.py \
              docs/__init__.py docs/overrides docs/docs/navigation_template.txt \
              docs/site
  git commit -m "build(docs): remove MkDocs deps and orchestration"
  ```

  `git add -A <pathspec>` stages adds, modifications, and removals for the
  given paths — necessary here because most of these are deletions.

---

### Task 7: Move content from `docs/docs/` up to `docs/`

**Files:**
- Move: `docs/docs/*` → `docs/*`
- Modify: `docs/conf.py` (remove the `if _SOURCE_SUFFIX_DIR.is_dir()` shim; remove `index_sphinx.md` exclude)
- Modify: `tools/build-docs.sh` (drop the `if [[ -d docs/docs ]]` branch)
- Modify: `tools/serve-docs.sh` (drop the `if [[ -d docs/docs ]]` branch)
- Delete: `docs/index.md` (the original MkDocs landing page)
- Rename: `docs/docs/index_sphinx.md` → `docs/index.md` (after content move)

- [ ] **Step 1: Move tracked content one level up**

  ```bash
  git mv docs/docs/about docs/about
  git mv docs/docs/benchmarks docs/benchmarks
  git mv docs/docs/concepts docs/concepts
  git mv docs/docs/guides docs/guides
  git mv docs/docs/contributing.md docs/contributing.md
  ```

- [ ] **Step 2: Replace the placeholder `docs/index.md` (if it exists from the original MkDocs setup) and the temporary `index_sphinx.md`**

  Check whether `docs/docs/index.md` was already moved or is still there:
  ```bash
  ls docs/docs/ 2>/dev/null
  ```
  If `docs/docs/index.md` still exists, replace it with `index_sphinx.md`:
  ```bash
  rm docs/docs/index.md
  git mv docs/docs/index_sphinx.md docs/index.md
  ```
  Otherwise (already gone), just:
  ```bash
  git mv docs/docs/index_sphinx.md docs/index.md
  ```

- [ ] **Step 3: Delete `docs/docs/` and any leftover static-asset duplicates**

  ```bash
  rm -rf docs/docs/
  ```

  (At this point the static assets at `docs/docs/stylesheets/`, `docs/docs/javascripts/`, `docs/docs/assets/` are also removed; their copies live at `docs/_static/`.)

- [ ] **Step 4: Update `docs/conf.py` to drop the source-directory shim**

  Open `docs/conf.py`. Remove these lines (added in Task 2 / Task 4):

  ```python
  # NOTE: Phase 1–2 invocation:
  #   sphinx-build -c docs docs/docs docs/_build/html
  # After Task 7 (content moves up), invocation becomes:
  #   sphinx-build docs docs/_build/html
  ```

  Replace the block:
  ```python
  _HERE = Path(__file__).resolve().parent
  _SOURCE_SUFFIX_DIR = _HERE / "docs"
  if _SOURCE_SUFFIX_DIR.is_dir():
      # Phase 1–2: point at docs/docs/
      master_doc = "index_sphinx"
  ```
  with:
  ```python
  master_doc = "index"
  ```

  Remove the temporary line from `exclude_patterns`:
  ```python
      "index.md",  # MkDocs landing page; Phase 1 uses index_sphinx.md
  ```

  Also remove the temporary `suppress_warnings` line if it was added in Task 4 step 6 (cross-ref warnings will be addressed in Task 8).

- [ ] **Step 5: Update `tools/build-docs.sh`**

  Replace the content with the simplified post-migration version:

  ```bash
  #!/usr/bin/env bash
  set -e
  set -x

  uv run sphinx-build -W docs docs/_build/html
  ```

- [ ] **Step 6: Update `tools/serve-docs.sh`**

  Replace the content with:

  ```bash
  #!/usr/bin/env bash
  set -e
  set -x

  uv run sphinx-autobuild docs docs/_build/html \
    --port 8008 --host 0.0.0.0 \
    --watch mononet
  ```

- [ ] **Step 7: Run the build (expect warnings about broken cross-references — fix in Task 8)**

  Run: `uv run sphinx-build docs docs/_build/html` (without `-W`)
  Expected: build completes; warnings appear about `../api/...` links in guides and concepts that don't resolve.

  Run without `-W` to confirm the build at least succeeds in this transitional state. Task 8 will make it `-W` clean.

- [ ] **Step 8: Temporarily relax `-W` so this commit's pre-commit hook passes**

  Pre-commit invokes `tools/build-docs.sh` which currently uses `-W`. The
  cross-ref warnings from the content move would fail the hook. We restore `-W`
  in Task 8 step 4 after fixing the cross-refs.

  Edit `tools/build-docs.sh`. Change:
  ```bash
  uv run sphinx-build -W docs docs/_build/html
  ```
  to:
  ```bash
  uv run sphinx-build docs docs/_build/html
  ```

  Verify the build now passes:
  Run: `./tools/build-docs.sh`
  Expected: builds; warnings printed but exit 0.

- [ ] **Step 9: Commit the move**

  ```bash
  git add -A docs tools/build-docs.sh tools/serve-docs.sh
  git commit -m "build(docs): move content from docs/docs/ to docs/"
  ```

---

### Task 8: Fix cross-references in existing markdown

The guides reference API symbols via relative paths like `[`MonoLinear`](../api/mononet/torch/MonoLinear.md)`. Convert these to MyST cross-references that point at the `sphinx-autodoc2`-generated pages.

**Files:**
- Modify: `docs/guides/pytorch.md`
- Modify: `docs/guides/jax.md`
- Modify: `docs/guides/keras.md`
- Modify: `docs/concepts/monotonicity.md`
- Modify: `docs/concepts/layers.md`
- Modify: `docs/index.md`
- Modify: `tools/build-docs.sh` (restore `-W`)

- [ ] **Step 1: Identify all broken cross-references**

  Run: `grep -rn '\.\./api/' docs/*.md docs/**/*.md 2>/dev/null`
  Expected: a list of lines like `[`MonoLinear`](../api/mononet/torch/MonoLinear.md)` in the guides.

- [ ] **Step 2: Inspect the actual generated API page slugs**

  Run: `ls docs/_build/html/apidocs/`
  Expected: pages like `mononet.torch.MonoLinear.html`, `mononet.jax.MonoLinear.html`, `mononet.MonotonicityMask.html`, etc. (autodoc2 flat-naming convention).

  Run: `ls docs/apidocs/ 2>/dev/null`
  Expected: source `.md` files mirror the same naming.

- [ ] **Step 3: Convert each link**

  For every `../api/<module path>/<symbol>.md` link, rewrite to a MyST cross-reference using the Python role. The MyST + autodoc2 form is:

  ```markdown
  {py:class}`mononet.torch.MonoLinear`
  ```
  or with custom display text:
  ```markdown
  [`MonoLinear`](#mononet.torch.MonoLinear)
  ```

  Example fix in `docs/guides/pytorch.md`. **Before:**
  ```markdown
  - [`MonoLinear`](../api/mononet/torch/MonoLinear.md) — monotonic
    analogue of `torch.nn.Linear`.
  - [`MonoMLP`](../api/mononet/torch/MonoMLP.md) — multi-layer composition.
  ```
  **After:**
  ```markdown
  - {py:class}`mononet.torch.MonoLinear` — monotonic analogue of
    {py:class}`torch.nn.Linear`.
  - {py:class}`mononet.torch.MonoMLP` — multi-layer composition.
  ```

  The `torch.nn.Linear` reference works via `intersphinx_mapping["torch"]`.

  Repeat for `docs/guides/jax.md` (`mononet.jax.MonoLinear`, `mononet.jax.MonoMLP`), `docs/guides/keras.md` (similarly, if it has any), `docs/concepts/monotonicity.md` (`mononet.core.types.MonotonicityMask`), `docs/concepts/layers.md`, and `docs/index.md`.

  Also convert plain-markdown cross-page links like `[PyTorch guide](guides/pytorch.md)` — these are fine as-is in MyST (it rewrites to `.html`); only the API links need restructuring.

- [ ] **Step 4: Restore `-W` in `tools/build-docs.sh`**

  Edit `tools/build-docs.sh`. Change:
  ```bash
  uv run sphinx-build docs docs/_build/html
  ```
  back to:
  ```bash
  uv run sphinx-build -W docs docs/_build/html
  ```

- [ ] **Step 5: Run the build with warnings-as-errors**

  Run: `./tools/build-docs.sh`
  Expected: passes with no warnings.

  If any cross-ref still fails: re-grep for `../api/` (Step 1) — that pattern must produce zero matches. Also check `_build/html` for any visible "broken link" rendering on the API pages.

- [ ] **Step 6: Spot-check the rendered HTML**

  Run: `uv run python -m http.server -d docs/_build/html 8009 &`
  Open `http://localhost:8009/guides/pytorch.html` and confirm:
  - The `MonoLinear` link is clickable.
  - It navigates to the autodoc2-generated `mononet.torch.MonoLinear` page.
  - That page shows the docstring + signature.

  Kill the server.

- [ ] **Step 7: Commit**

  ```bash
  git add docs/ tools/build-docs.sh
  git commit -m "docs: convert API cross-references to MyST py-class refs"
  ```

---

## Phase 4 — Polish

---

### Task 9: Replace `overrides/main.html` social/OG meta tags

The old `docs/overrides/main.html` injected OG/Twitter meta tags. PyData Sphinx Theme exposes a `layout.html` override mechanism.

**Files:**
- Create: `docs/_templates/layout.html`

- [ ] **Step 1: Write the override template**

  Write the following to `docs/_templates/layout.html`:

  ```jinja
  {% extends "!layout.html" %}

  {% block extrahead %}
    {{ super() }}
    {% set site_image = "https://opengraph.githubassets.com/1671805243.560327/davorrunje/mononet" %}
    {% set site_description = "Unconstrained monotonic neural networks for PyTorch, JAX, and Keras" %}
    <meta property="og:type" content="website" />
    <meta property="og:title" content="{{ project }}{% if title and title != project %} — {{ title }}{% endif %}" />
    <meta property="og:description" content="{{ site_description }}" />
    <meta property="og:url" content="{{ pageurl|default(html_baseurl|default('')) }}" />
    <meta property="og:image" content="{{ site_image }}" />
    <meta property="og:image:type" content="image/png" />
    <meta property="og:image:width" content="1200" />
    <meta property="og:image:height" content="630" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="{{ project }}{% if title and title != project %} — {{ title }}{% endif %}" />
    <meta name="twitter:description" content="{{ site_description }}" />
    <meta name="twitter:image" content="{{ site_image }}" />
  {% endblock %}
  ```

- [ ] **Step 2: Remove the `_templates/.gitkeep` if it still exists**

  ```bash
  rm -f docs/_templates/.gitkeep
  ```

- [ ] **Step 3: Build and verify the tags are emitted**

  Run: `./tools/build-docs.sh`
  Expected: passes.

  Run: `grep 'og:title' docs/_build/html/index.html`
  Expected: a line containing `<meta property="og:title" content="mononet" />`.

- [ ] **Step 4: Commit**

  ```bash
  git add docs/_templates/layout.html
  git commit -m "docs: port OG/Twitter meta tag overrides to Sphinx layout.html"
  ```

---

### Task 10: Smoke-test math rendering and intersphinx

**Files:**
- Create: `docs/_smoke.md` (temporary; removed at end of task)

- [ ] **Step 1: Write a temporary smoke-test page**

  Write the following to `docs/_smoke.md`:

  ````markdown
  ---
  orphan: true
  ---

  # Smoke test

  Inline math: $f: \mathbb{R}^n \to \mathbb{R}$.

  Block math:

  $$
  \frac{\partial f}{\partial x_i} \geq 0 \quad \forall i \in S^+
  $$

  AMS environment:

  ```{math}
  :label: monotonicity
  y = \sigma(W^+ x_{S^+} + W^- x_{S^-} + b)
  ```

  Equation reference: {eq}`monotonicity`.

  Intersphinx test: link to {py:class}`torch.Tensor` and {py:class}`numpy.ndarray`.
  ````

  Note `orphan: true` keeps Sphinx from complaining the page isn't in a toctree.

- [ ] **Step 2: Build and inspect**

  Run: `./tools/build-docs.sh`
  Expected: passes with no warnings.

  Run: `grep -q 'MathJax' docs/_build/html/_smoke.html && echo MATH_OK`
  Expected: `MATH_OK`.

  Run: `grep -q 'href="https://pytorch.org/docs/stable' docs/_build/html/_smoke.html && echo INTERSPHINX_OK`
  Expected: `INTERSPHINX_OK` (the `torch.Tensor` reference resolves through intersphinx to the PyTorch docs).

- [ ] **Step 3: Remove the smoke-test page**

  ```bash
  rm docs/_smoke.md
  ```

- [ ] **Step 4: Re-build to confirm clean state**

  Run: `./tools/build-docs.sh`
  Expected: still passes.

- [ ] **Step 5: Commit**

  No new files to add (smoke test was temporary). If `rm docs/_smoke.md` produced an unstaged delete, ensure the working tree is clean:
  ```bash
  git status
  ```
  Expected: clean working tree.

  No commit needed for this task — it's verification only.

---

## Phase 5 — Versioning and CI

---

### Task 11: Add versions.json generator

`sphinx-multiversion` builds separate per-version output directories but doesn't emit a `versions.json`. The PyData theme's version switcher needs one.

**Files:**
- Create: `tools/gen_versions_json.py`

- [ ] **Step 1: Write the generator script**

  Write the following to `tools/gen_versions_json.py`:

  ```python
  """Generate versions.json for the PyData Sphinx Theme version switcher.

  Run after `sphinx-multiversion` has populated docs/_build/html/<version>/
  directories. Writes versions.json at the root of the build output.

  Args:
      build_dir: path to the multiversion build output
          (e.g. docs/_build/html).
      base_url: the site's base URL (used to construct per-version URLs).
  """

  from __future__ import annotations

  import argparse
  import json
  import re
  from pathlib import Path

  VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


  def main() -> None:
      parser = argparse.ArgumentParser()
      parser.add_argument("build_dir", type=Path)
      parser.add_argument("base_url", type=str)
      args = parser.parse_args()

      base = args.base_url.rstrip("/")
      entries: list[dict[str, str]] = []

      version_dirs = sorted(
          p.name for p in args.build_dir.iterdir() if p.is_dir()
      )
      tagged = [v for v in version_dirs if VERSION_RE.match(v)]
      tagged.sort(reverse=True)  # newest first

      if "main" in version_dirs:
          entries.append({
              "name": "dev (main)",
              "version": "latest",
              "url": f"{base}/main/",
          })

      for v in tagged:
          entry: dict[str, str] = {
              "name": v,
              "version": v,
              "url": f"{base}/{v}/",
          }
          if v == tagged[0]:
              entry["preferred"] = "true"
          entries.append(entry)

      out = args.build_dir / "versions.json"
      out.write_text(json.dumps(entries, indent=2) + "\n")
      print(f"wrote {out}")


  if __name__ == "__main__":
      main()
  ```

- [ ] **Step 2: Test the script against a single-version build**

  Run: `./tools/build-docs.sh`
  (this produces `docs/_build/html/` with content at the root, not under per-version dirs)

  Run: `uv run python tools/gen_versions_json.py docs/_build/html https://davorrunje.github.io/mononet`
  Expected: prints `wrote docs/_build/html/versions.json` and the file contains `[]` (empty array — no per-version subdirs found). This is the correct behavior for single-version builds.

- [ ] **Step 3: Test against a simulated multiversion layout**

  ```bash
  mkdir -p docs/_build/html/main docs/_build/html/v0.1.0
  uv run python tools/gen_versions_json.py docs/_build/html https://davorrunje.github.io/mononet
  cat docs/_build/html/versions.json
  ```

  Expected output:
  ```json
  [
    {
      "name": "dev (main)",
      "version": "latest",
      "url": "https://davorrunje.github.io/mononet/main/"
    },
    {
      "name": "v0.1.0",
      "version": "v0.1.0",
      "url": "https://davorrunje.github.io/mononet/v0.1.0/",
      "preferred": "true"
    }
  ]
  ```

  Clean up:
  ```bash
  rm -rf docs/_build/html/main docs/_build/html/v0.1.0 docs/_build/html/versions.json
  ```

- [ ] **Step 4: Verify it lints clean**

  Run: `uv run ruff check tools/gen_versions_json.py`
  Expected: passes.

  Run: `uv run mypy tools/gen_versions_json.py`
  Expected: passes.

- [ ] **Step 5: Commit**

  ```bash
  git add tools/gen_versions_json.py
  git commit -m "build(docs): add versions.json generator for sphinx-multiversion"
  ```

---

### Task 12: Update GitHub Actions docs workflow

**Files:**
- Modify: `.github/workflows/docs.yml`

- [ ] **Step 1: Replace the workflow**

  Replace the entire content of `.github/workflows/docs.yml` with:

  ```yaml
  name: Docs

  on:
    push:
      branches: [main]
      tags: ['v*.*.*']
    workflow_dispatch:

  permissions:
    contents: write

  concurrency:
    group: docs-${{ github.ref }}
    cancel-in-progress: true

  jobs:
    deploy:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v6
          with:
            fetch-depth: 0
            # required so sphinx-multiversion can see all branches/tags
            fetch-tags: true

        - uses: actions/setup-python@v6
          with:
            python-version: "3.13"

        - uses: astral-sh/setup-uv@v7

        - name: Install dependencies
          run: uv sync --group docs --extra all

        - name: Build all versions
          run: |
            git config user.name  "github-actions[bot]"
            git config user.email "github-actions[bot]@users.noreply.github.com"
            uv run sphinx-multiversion -W docs docs/_build/html

        - name: Generate versions.json
          run: |
            uv run python tools/gen_versions_json.py \
              docs/_build/html https://davorrunje.github.io/mononet

        - name: Write a root index.html that redirects to /main/
          run: |
            cat > docs/_build/html/index.html <<'EOF'
            <!doctype html>
            <meta charset="utf-8">
            <meta http-equiv="refresh" content="0; url=./main/">
            <link rel="canonical" href="./main/">
            EOF

        - name: Deploy to gh-pages
          uses: peaceiris/actions-gh-pages@v4
          with:
            github_token: ${{ secrets.GITHUB_TOKEN }}
            publish_dir: docs/_build/html
            publish_branch: gh-pages
            keep_files: false
            commit_message: "Deploy docs from ${{ github.sha }}"
  ```

- [ ] **Step 2: Validate the workflow YAML**

  Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/docs.yml'))"`
  Expected: no output, exit 0.

  Run: `uv run pre-commit run check-yaml --files .github/workflows/docs.yml`
  Expected: passes.

- [ ] **Step 3: Trial-run the build step locally (simulates what the workflow will do)**

  Run:
  ```bash
  uv sync --group docs --extra all
  uv run sphinx-multiversion -W docs docs/_build/html
  ```

  Expected: builds successfully for `main`. If on a fresh checkout with no remote refs configured, `sphinx-multiversion` may build nothing — this is fine; the workflow runs with `fetch-tags: true`.

  If it fails locally with "could not find any branches matching whitelist", that's expected outside CI. The CI step exercises the real path.

- [ ] **Step 4: Commit**

  ```bash
  git add .github/workflows/docs.yml
  git commit -m "ci(docs): switch GitHub Actions to Sphinx multiversion build"
  ```

---

## Phase 6 — Documentation update

---

### Task 13: Update CLAUDE.md docs commands section

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update the Documentation section**

  Open `CLAUDE.md`. Find the section:

  ```markdown
  ### Documentation
  ```bash
  ./tools/build-docs.sh                                # build docs
  ./tools/serve-docs.sh                                # live preview
  ```
  ```

  No change needed to the commands themselves (the scripts still exist). But add a line about the engine:

  ```markdown
  ### Documentation
  Built with Sphinx + PyData Sphinx Theme. See `docs/conf.py`.

  ```bash
  ./tools/build-docs.sh                                # build docs (sphinx-build -W)
  ./tools/serve-docs.sh                                # live preview (sphinx-autobuild)
  uv run sphinx-multiversion -W docs docs/_build/html  # build all versions (CI uses this)
  ```
  ```

- [ ] **Step 2: Verify final build cleanly**

  Run: `./tools/build-docs.sh`
  Expected: `-W` clean.

  Run: `uv run pre-commit run --all-files`
  Expected: all hooks pass.

  Run: `uv run pytest -q`
  Expected: passes (docs migration shouldn't touch package code).

  Run: `uv run mypy`
  Expected: passes.

- [ ] **Step 3: Commit**

  ```bash
  git add CLAUDE.md
  git commit -m "docs(claude): document Sphinx as the docs engine"
  ```

---

## Final verification checklist

Before merging, run through every item in the spec's "Testing & verification" section. Concretely:

- [ ] `uv run sphinx-build -W docs docs/_build/html` exits 0 with no warnings.
- [ ] `./tools/build-docs.sh` succeeds and emits no MkDocs ProperDocs warning.
- [ ] `./tools/serve-docs.sh` serves on `http://0.0.0.0:8008` (test: `curl -fsS http://0.0.0.0:8008/ -o /dev/null` after starting the server in a separate terminal, then kill).
- [ ] Every page in the spec's content parity list renders (`index.html`, all `guides/`, all `concepts/`, `benchmarks/index.html`, `benchmarks/00-overview.html`, every `apidocs/mononet.*.html` page, `about/*.html`, `contributing.html`).
- [ ] API reference covers `mononet`, `mononet.core`, `mononet.jax`, `mononet.torch` — verified by listing `docs/_build/html/apidocs/`.
- [ ] Code blocks have copy buttons (`sphinx-copybutton`).
- [ ] Light/dark mode toggle works (manual click in browser).
- [ ] Edit-on-GitHub link present on each page (`use_edit_page_button: True`).
- [ ] Search box works (Sphinx built-in).
- [ ] Math renders (`$\mathbb{R}^n$` in any page).
- [ ] `torch.Tensor` references resolve to PyTorch docs via intersphinx.
- [ ] `uv run pytest`, `uv run mypy`, `uv run ruff check`, `uv run pre-commit run --all-files` all pass.
- [ ] No file references to `mkdocs.yml`, `mkdocstrings`, `mkdocs-material`, `mike`, `create_api_docs.py`, `docs.py`, `overrides/` remain in the repo (`git grep mkdocs` should return only the spec/plan documents in `docs/superpowers/`).
