# MyST Docstring Conversion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the 4 docstrings that use Google-style `Args:`/`Returns:` blocks to MyST field-list style (`:param:` / `:returns:` / `:raises:`), update tooling/docs to reflect the new convention, and verify the rendered API pages show structured argument tables instead of flat prose.

**Architecture:** Six small, focused commits. Task 1 relaxes the ruff convention first (so old-style docstrings still lint clean during the transition). Tasks 2–4 convert one source file each. Tasks 5–6 update project docs and remove the now-obsolete TODO comment. Each commit leaves the pre-commit `docs` hook green.

**Tech Stack:** Sphinx + `sphinx-autodoc2` + MyST `fieldlist` extension (already enabled in `docs/conf.py`). Ruff `pep257` pydocstyle convention. Python 3.13.

**Spec:** [docs/superpowers/specs/2026-05-22-myst-docstrings-design.md](../specs/2026-05-22-myst-docstrings-design.md)

---

## File map

**Modified:**
- `pyproject.toml` — Task 1 (one-line convention change)
- `mononet/core/reference.py` — Task 2 (2 docstrings)
- `mononet/torch/layers.py` — Task 3 (1 class docstring)
- `tools/gen_versions_json.py` — Task 4 (module docstring)
- `CLAUDE.md` — Task 5 (single line update)
- `CONTRIBUTING.md` — Task 5 (single line update)
- `docs/conf.py` — Task 6 (remove TODO comment block)

**Created/deleted:** None.

---

### Task 1: Switch ruff pydocstyle convention from "google" to "pep257"

This task **must run first**. The current ruff `convention = "google"` enables Google-specific D-rules that would flag MyST `:param:` blocks once we convert. Switching to `"pep257"` first means both the existing Google docstrings AND the new MyST field-list docstrings pass — so the build stays green throughout the conversion.

**Files:**
- Modify: `pyproject.toml:229-230`

- [ ] **Step 1: Update the pydocstyle convention**

  Use the Edit tool on `pyproject.toml`. Find:

  ```toml
  [tool.ruff.lint.pydocstyle]
  convention = "google"
  ```

  Replace with:

  ```toml
  [tool.ruff.lint.pydocstyle]
  convention = "pep257"
  ```

- [ ] **Step 2: Confirm ruff still passes on the existing (Google-style) docstrings**

  Run: `uv run ruff check`
  Expected: `All checks passed!` — `pep257` is strictly more permissive than `google` for the existing docstrings, so nothing should newly fail.

- [ ] **Step 3: Confirm the Sphinx build still passes**

  Run: `./tools/build-docs.sh`
  Expected: exit 0 (`-W` clean). Docstring rendering doesn't change for this commit — the visible output is identical.

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml
  git commit -m "build(lint): switch ruff pydocstyle convention to pep257"
  ```

  Add the standard `Co-Authored-By` trailer used in other commits in this repo (check `git log -3 --format=%B` for the format).

---

### Task 2: Convert docstrings in `mononet/core/reference.py`

Convert both function docstrings (`monotonic_dense`, `monotonic_mlp`) from Google-style to MyST field-list. Both have the same shape (`x`, `weights`, `bias`, `mask`, `activation` params; `Returns:` block).

**Files:**
- Modify: `mononet/core/reference.py:23-44, 47-68`

- [ ] **Step 1: Convert `monotonic_dense` docstring**

  Use the Edit tool on `mononet/core/reference.py`. Find:

  ```python
      """Single-layer monotonic transformation (NumPy reference).

      Args:
          x: Input array of shape (batch, in_features).
          weights: Unconstrained weights of shape (in_features, out_features).
          bias: Bias vector of shape (out_features,).
          mask: Per-input monotonicity mask.
          activation: Activation specification.

      Returns:
          Output array of shape (batch, out_features).
      """
  ```

  Replace with:

  ```python
      """Single-layer monotonic transformation (NumPy reference).

      :param x: Input array of shape `(batch, in_features)`.
      :param weights: Unconstrained weights of shape
          `(in_features, out_features)`.
      :param bias: Bias vector of shape `(out_features,)`.
      :param mask: Per-input monotonicity mask.
      :param activation: Activation specification.
      :returns: Output array of shape `(batch, out_features)`.
      """
  ```

  (Shape tuples are wrapped in inline code spans for readability — this is the
  preferred style per the spec.)

- [ ] **Step 2: Convert `monotonic_mlp` docstring**

  Find (in the same file):

  ```python
      """Multi-layer monotonic MLP (NumPy reference).

      Args:
          x: Input array of shape (batch, in_features).
          weights: Per-layer weight arrays.
          biases: Per-layer bias vectors.
          mask: Monotonicity mask applied to the first layer.
          activation: Activation used between hidden layers.

      Returns:
          Output array of shape (batch, weights[-1].shape[1]).
      """
  ```

  Replace with:

  ```python
      """Multi-layer monotonic MLP (NumPy reference).

      :param x: Input array of shape `(batch, in_features)`.
      :param weights: Per-layer weight arrays.
      :param biases: Per-layer bias vectors.
      :param mask: Monotonicity mask applied to the first layer.
      :param activation: Activation used between hidden layers.
      :returns: Output array of shape `(batch, weights[-1].shape[1])`.
      """
  ```

- [ ] **Step 3: Confirm ruff and mypy pass**

  Run: `uv run ruff check`
  Expected: `All checks passed!`

  Run: `uv run mypy`
  Expected: `Success: no issues found in N source files`

- [ ] **Step 4: Confirm the Sphinx build is `-W` clean**

  Run: `./tools/build-docs.sh`
  Expected: exit 0 with no warnings.

- [ ] **Step 5: Verify the rendered HTML shows a structured field list**

  Open `docs/_build/html/apidocs/mononet.core.reference.html` (or use a grep
  check):

  Run: `grep -A2 'field-list' docs/_build/html/apidocs/mononet.core.reference.html | head -20`
  Expected: lines containing `<dl class="field-list">` and individual `<dt>` /
  `<dd>` pairs for each parameter — proves the field-list rendered structurally.

  If the grep shows `<dl class="field-list">` but the `<dt>` content looks
  empty, the MyST `fieldlist` extension may not be parsing correctly — stop
  and investigate (this would indicate a deeper config issue, not a content
  issue).

- [ ] **Step 6: Commit**

  ```bash
  git add mononet/core/reference.py
  git commit -m "docs(reference): convert docstrings to MyST field-list"
  ```

  Add the `Co-Authored-By` trailer.

---

### Task 3: Convert the `MonoLinear` class docstring in `mononet/torch/layers.py`

This is a class docstring (not a function), with 4 parameters. The
`__init__` and `forward` method docstrings are summary-only (no parameter
blocks) and don't need conversion.

**Files:**
- Modify: `mononet/torch/layers.py:14-22`

- [ ] **Step 1: Convert the `MonoLinear` class docstring**

  Use the Edit tool on `mononet/torch/layers.py`. Find:

  ```python
  class MonoLinear(nn.Module):
      """Monotonic analogue of `torch.nn.Linear`.

      Args:
          in_features: Number of input features.
          out_features: Number of output features.
          monotonicity: Per-input-feature monotonicity mask.
          activation: Activation specification (resolved by the kernel).
      """
  ```

  Replace with:

  ```python
  class MonoLinear(nn.Module):
      """Monotonic analogue of `torch.nn.Linear`.

      :param in_features: Number of input features.
      :param out_features: Number of output features.
      :param monotonicity: Per-input-feature monotonicity mask.
      :param activation: Activation specification (resolved by the kernel).
      """
  ```

- [ ] **Step 2: Confirm ruff and mypy pass**

  Run: `uv run ruff check`
  Expected: `All checks passed!`

  Run: `uv run mypy`
  Expected: `Success: no issues found`

- [ ] **Step 3: Confirm the Sphinx build is `-W` clean**

  Run: `./tools/build-docs.sh`
  Expected: exit 0 with no warnings.

- [ ] **Step 4: Verify rendered HTML**

  Run: `grep -A2 'field-list' docs/_build/html/apidocs/mononet.torch.layers.html | head -10`
  Expected: `<dl class="field-list">` with the four `<dt>` entries for the
  parameters.

- [ ] **Step 5: Commit**

  ```bash
  git add mononet/torch/layers.py
  git commit -m "docs(torch): convert MonoLinear docstring to MyST field-list"
  ```

  Add the `Co-Authored-By` trailer.

---

### Task 4: Convert the module docstring in `tools/gen_versions_json.py`

This is a module-level docstring. It has an `Args:` block but no parameters
to the *module* itself — the original docstring documented the script's
command-line arguments. The new docstring should keep the same semantic but
use field-list syntax.

**Files:**
- Modify: `tools/gen_versions_json.py:1-10`

- [ ] **Step 1: Convert the module docstring**

  Use the Edit tool on `tools/gen_versions_json.py`. Find:

  ```python
  """Generate versions.json for the PyData Sphinx Theme version switcher.

  Run after `sphinx-multiversion` has populated docs/_build/html/<version>/
  directories. Writes versions.json at the root of the build output.

  Args:
      build_dir: path to the multiversion build output
          (e.g. docs/_build/html).
      base_url: the site's base URL (used to construct per-version URLs).
  """
  ```

  Replace with:

  ```python
  """Generate versions.json for the PyData Sphinx Theme version switcher.

  Run after `sphinx-multiversion` has populated docs/_build/html/<version>/
  directories. Writes versions.json at the root of the build output.

  :param build_dir: Path to the multiversion build output
      (e.g. `docs/_build/html`).
  :param base_url: The site's base URL (used to construct per-version URLs).
  """
  ```

  Note: this is a module docstring — `:param:` fields here document the CLI
  arguments to `main()`. Strictly, module docstrings don't have parameters,
  but ruff `pep257` doesn't enforce that and this matches the prior author's
  intent. Acceptable.

- [ ] **Step 2: Confirm ruff and mypy pass**

  Run: `uv run ruff check`
  Expected: `All checks passed!`

  Run: `uv run mypy`
  Expected: `Success: no issues found`

- [ ] **Step 3: Confirm the Sphinx build is `-W` clean**

  Run: `./tools/build-docs.sh`
  Expected: exit 0.

- [ ] **Step 4: Commit**

  ```bash
  git add tools/gen_versions_json.py
  git commit -m "docs(tools): convert gen_versions_json docstring to MyST field-list"
  ```

  Add the `Co-Authored-By` trailer.

---

### Task 5: Update `CLAUDE.md` and `CONTRIBUTING.md`

Update the two project-level convention documents that mention Google-style
docstrings.

**Files:**
- Modify: `CLAUDE.md:81`
- Modify: `CONTRIBUTING.md:116`

- [ ] **Step 1: Update `CLAUDE.md`**

  Use the Edit tool on `CLAUDE.md`. Find:

  ```markdown
  - Google-style docstrings
  ```

  Replace with:

  ```markdown
  - MyST field-list docstrings: `:param x: ...` / `:returns: ...` / `:raises X: ...`. Types come from signature annotations, never `:type:`/`:rtype:`. Body text is MyST markdown.
  ```

- [ ] **Step 2: Update `CONTRIBUTING.md`**

  Use the Edit tool on `CONTRIBUTING.md`. Find:

  ```markdown
  - Google-style docstrings on all public functions and classes.
  ```

  Replace with:

  ```markdown
  - MyST field-list docstrings on all public functions and classes (`:param x: ...`, `:returns: ...`, `:raises X: ...`). Types come from signature annotations, never `:type:`/`:rtype:`. See [the spec](docs/superpowers/specs/2026-05-22-myst-docstrings-design.md) for the canonical format.
  ```

- [ ] **Step 3: Confirm both edits applied correctly**

  Run: `grep -n -i 'google.*docstring\|docstring.*google' CLAUDE.md CONTRIBUTING.md`
  Expected: no matches.

  Run: `grep -n 'MyST field-list' CLAUDE.md CONTRIBUTING.md`
  Expected: two matches (one per file).

- [ ] **Step 4: Confirm pre-commit hooks pass on the doc changes**

  Run: `uv run pre-commit run --files CLAUDE.md CONTRIBUTING.md`
  Expected: all hooks pass (these files trigger trim-whitespace, eof-fixer,
  codespell, detect-secrets).

- [ ] **Step 5: Commit**

  ```bash
  git add CLAUDE.md CONTRIBUTING.md
  git commit -m "docs: update convention guides to reference MyST field-list docstrings"
  ```

  Add the `Co-Authored-By` trailer.

---

### Task 6: Remove the obsolete TODO comment in `docs/conf.py` and run final verification

The TODO comment at `docs/conf.py:87-90` documents a limitation that no
longer applies (because the docstrings are now MyST-native). Remove it as
the last step of the migration.

**Files:**
- Modify: `docs/conf.py:87-90`

- [ ] **Step 1: Remove the TODO comment block**

  Use the Edit tool on `docs/conf.py`. Find:

  ```python
  autodoc2_render_plugin = "myst"
  # TODO: autodoc2 has no Google-style docstring parser; "myst" parses Google
  # sections as flat prose. Args:/Returns: blocks won't render structured
  # until a griffe/napoleon-based parser is added (or src docstrings rewrite).
  # Spec calls for Google-style preservation — visual fidelity lost for now.
  autodoc2_docstring_parser_regexes = [
      (r".*", "myst"),
  ]
  ```

  Replace with:

  ```python
  autodoc2_render_plugin = "myst"
  autodoc2_docstring_parser_regexes = [
      (r".*", "myst"),
  ]
  ```

  (The four comment lines are removed; nothing else changes.)

- [ ] **Step 2: Confirm no `Args:`/`Returns:`/`Yields:`/`Raises:` blocks remain in source**

  Run: `grep -rn 'Args:\|Returns:\|Yields:\|Raises:' mononet/ tools/ --include='*.py'`
  Expected: no matches.

- [ ] **Step 3: Confirm new field-list syntax IS present in source**

  Run: `grep -rn ':param \|:returns:\|:raises ' mononet/ tools/ --include='*.py' | wc -l`
  Expected: at least 18 matches (12 in `reference.py` — 5 params + 1 returns
  per function, 2 functions; 4 in `layers.py` — 4 params on `MonoLinear`;
  2 in `gen_versions_json.py` — 2 params).

- [ ] **Step 4: Final full verification**

  Run: `./tools/build-docs.sh`
  Expected: exit 0, `-W` clean.

  Run: `uv run ruff check`
  Expected: `All checks passed!`

  Run: `uv run mypy`
  Expected: `Success: no issues found`

  Run: `uv run pytest -q`
  Expected: passes (unchanged — docstring-only changes don't affect tests).

  Run: `uv run pre-commit run --all-files`
  Expected: all hooks pass.

- [ ] **Step 5: Spot-check one rendered API page**

  Run: `grep -c 'field-list' docs/_build/html/apidocs/mononet.core.reference.html docs/_build/html/apidocs/mononet.torch.layers.html`
  Expected: at least 1 occurrence per file (proves the field list rendered).

  Run: `grep -i 'Args:' docs/_build/html/apidocs/mononet.core.reference.html`
  Expected: NO matches — confirms the literal `Args:` text from the old
  docstrings is not in the rendered output.

- [ ] **Step 6: Commit**

  ```bash
  git add docs/conf.py
  git commit -m "docs(conf): remove obsolete Google-style docstring TODO"
  ```

  Add the `Co-Authored-By` trailer.

---

## Final state checklist

After Task 6:

- [ ] No `Args:` / `Returns:` / `Yields:` / `Raises:` blocks remain in `mononet/` or `tools/` Python sources.
- [ ] All previously Google-style docstrings (4 total) now use `:param:` / `:returns:` field-list syntax.
- [ ] `pyproject.toml` has `convention = "pep257"`.
- [ ] `CLAUDE.md` and `CONTRIBUTING.md` describe the MyST field-list convention.
- [ ] `docs/conf.py` no longer contains the TODO comment about Google-style limitations.
- [ ] `uv run ruff check`, `uv run mypy`, `uv run pytest -q`, `./tools/build-docs.sh -W`, `uv run pre-commit run --all-files` all pass.
- [ ] The rendered API pages for `mononet.core.reference` and `mononet.torch.layers` show parameters in structured `<dl class="field-list">` tables rather than as flat prose paragraphs.
