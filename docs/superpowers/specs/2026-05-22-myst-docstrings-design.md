# Switch Docstring Convention to MyST Field-List — Design

**Status:** Draft for review
**Date:** 2026-05-22
**Owner:** Davor Runje
**Triggering event:** Follow-up to the
[Sphinx migration](2026-05-22-sphinx-migration-design.md). `sphinx-autodoc2`
has no Google-style docstring parser; `Args:` / `Returns:` blocks render as
flat prose in the generated API pages. The Sphinx-migration PR captured this
as a known TODO at [docs/conf.py:87-90](../../conf.py#L87-L90).

## Goal

Switch the project's docstring convention from Google-style to MyST field-list
style (`:param x: ...`, `:returns: ...`, `:raises X: ...`) so that
`sphinx-autodoc2` + `myst-nb` renders parameter/return/raise sections as
structured field-list HTML rather than flat prose.

## Non-goals

- Switching the API documentation generator (stays on `sphinx-autodoc2`).
- Switching to Numpy or reST docstring conventions.
- Writing a custom autodoc2 parser for Google-style.
- Adding `:type:` / `:rtype:` fields (signatures carry the types).
- Cross-referencing every type mention with `{py:class}` (optional polish).
- Touching docstrings that have no `Args:` / `Returns:` / `Raises:` block —
  those are already valid MyST prose.

## Background

Current state, as of 2026-05-22:

- `sphinx-autodoc2`'s `autodoc2_docstring_parser_regexes = [(r".*", "myst")]`
  parses each docstring as MyST markdown. The MyST `fieldlist` extension is
  enabled in [docs/conf.py](../../conf.py), so `:param x: ...` fields render
  as a structured HTML field list.
- 4 docstrings in the codebase still use Google-style sections (`Args:` /
  `Returns:`):
  - [mononet/core/reference.py](../../../mononet/core/reference.py) — 2 functions
    (`monotonic_dense`, `monotonic_mlp`).
  - [mononet/torch/layers.py](../../../mononet/torch/layers.py) — 1 class
    (`MonoLinear`).
  - [tools/gen_versions_json.py](../../../tools/gen_versions_json.py) — 1
    module docstring.
- 18 files in the `mononet/` package have docstrings. The other ~15 are
  module-level / class-level / function-summary docstrings without parameter
  blocks; they're already valid MyST.
- Ruff enforces `convention = "google"` in
  [pyproject.toml:229-230](../../../pyproject.toml#L229-L230). This will
  flag the new field-list style unless changed.
- [CLAUDE.md](../../../CLAUDE.md) explicitly mandates "Google-style docstrings".

Decisions taken during brainstorming:

1. **Format:** MyST field list (`:param x:` / `:returns:` / `:raises X:`).
   Rejected alternatives: definition list (less IDE-friendly), pure MyST
   prose (loses structure), keep Google + write custom parser (too much
   work for the visible benefit).
2. **Ruff pydocstyle:** switch `convention = "google"` to
   `convention = "pep257"`. Keeps presence and first-line format checks;
   drops the Google-specific section rules.

## Architecture

This change is content-and-config only. No code structure or build pipeline
changes.

### Format definition

```python
def func(x: int, y: str) -> bool:
    """One-line summary in the imperative mood.

    Optional paragraph of additional context. May span multiple lines and
    include MyST markdown (cross-refs like {py:class}`mononet.X`, math like
    $x^2$, inline `code`, etc.).

    :param x: Description of x. Type comes from the signature annotation.
    :param y: Description of y.
    :returns: Description of return value.
    :raises ValueError: when the input is invalid.
    """
```

Rules:

- **Summary line** stays unchanged (imperative, one sentence, ends in `.`).
- **`:param NAME: DESCRIPTION`** for each parameter.
- **`:returns:`** (not `:return:`) when there's a return value worth documenting.
- **`:raises EXCEPTION:`** for each documented exception.
- **No `:type:` / `:rtype:`** — type annotations are mandatory in this project
  (strict mypy), so duplicating them in the docstring is redundant.
- **Body text is MyST markdown** — cross-refs, math, inline code, links all work.

### File changes

| File | Change |
| --- | --- |
| [mononet/core/reference.py](../../../mononet/core/reference.py) | Convert 2 docstrings (`monotonic_dense`, `monotonic_mlp`). |
| [mononet/torch/layers.py](../../../mononet/torch/layers.py) | Convert 1 docstring (`MonoLinear`). |
| [tools/gen_versions_json.py](../../../tools/gen_versions_json.py) | Convert module docstring. |
| [pyproject.toml](../../../pyproject.toml) (line 229-230) | `convention = "google"` → `convention = "pep257"`. |
| [CLAUDE.md](../../../CLAUDE.md) (Code Style section) | "Google-style docstrings" → "MyST field-list docstrings (`:param x:` / `:returns:` / `:raises:`)". |
| [CONTRIBUTING.md](../../../CONTRIBUTING.md) (line 116) | "Google-style docstrings on all public functions and classes." → "MyST field-list docstrings on all public functions and classes (`:param x:` / `:returns:` / `:raises:`)." |
| [docs/conf.py](../../conf.py) (lines 87-90) | Remove the TODO comment block — no longer applies. |

### Example conversion

**Before** ([mononet/core/reference.py:23-44](../../../mononet/core/reference.py#L23-L44)):

```python
def monotonic_dense(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
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

**After:**

```python
def monotonic_dense(
    x: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
    bias: npt.NDArray[np.floating],
    mask: MonotonicityMask,
    activation: ActivationSpec,
) -> npt.NDArray[np.floating]:
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

Note: shape tuples are wrapped in inline code spans (`` `(...)` ``) in the
new format. This is optional but improves readability and consistency with
how shapes appear in MyST-rendered text. The original used plain prose for
shapes; the conversion can either preserve that or upgrade to code spans.
Preference: upgrade.

## Components

### Ruff configuration

In [pyproject.toml](../../../pyproject.toml) line 229-230:

```toml
[tool.ruff.lint.pydocstyle]
convention = "google"
```

Change to:

```toml
[tool.ruff.lint.pydocstyle]
convention = "pep257"
```

`pep257` is ruff's standard non-Google, non-Numpy convention. It keeps:
- D1xx: docstring presence rules.
- D2xx: docstring format rules (first-line ends in period, blank line after
  summary, etc.).

…and drops the Google-specific section-name and indentation rules that would
flag `:param x:` blocks.

After the change, `uv run ruff check` should pass on the converted files
without further changes.

### CLAUDE.md

In the "Code Style" section, the line "Google-style docstrings" becomes:

> MyST field-list docstrings: `:param x: …` for parameters, `:returns: …` for
> return values, `:raises X: …` for exceptions. Types are in signature
> annotations, never in docstring fields. Body text is MyST markdown
> (cross-refs, math, inline code).

### docs/conf.py

Remove these four lines (lines 87-90, the TODO comment block above
`autodoc2_docstring_parser_regexes`):

```python
# TODO: autodoc2 has no Google-style docstring parser; "myst" parses Google
# sections as flat prose. Args:/Returns: blocks won't render structured
# until a griffe/napoleon-based parser is added (or src docstrings rewrite).
# Spec calls for Google-style preservation — visual fidelity lost for now.
```

The `autodoc2_docstring_parser_regexes = [(r".*", "myst")]` line itself stays
— MyST is the correct parser for the new format. No other `conf.py` change.

## Data flow

```
source docstring (`:param x:` syntax)
        │
        ▼
sphinx-autodoc2 generates apidocs/*.md
        │
        ▼
MyST parser + fieldlist extension
        │
        ▼
docutils field_list node
        │
        ▼
HTML <dl class="field-list"> with proper structure
```

The MyST `fieldlist` extension (enabled in [docs/conf.py](../../conf.py)) is
what turns `:param x: ...` into a `field_list` docutils node. Without that
extension the fields would render as inline reST-look-alike text. The
extension is already on (it was enabled during the Sphinx migration), so no
configuration change is required.

## Error handling

Not applicable (no runtime code).

The only failure modes are at the linter/build level:

- **Ruff still flags some docstrings after the convention switch.** Mitigation:
  read the actual ruff output and adjust individual docstrings (e.g., the
  first-line-ends-in-period rule may catch a docstring that's been edited
  inconsistently).
- **Sphinx build emits a warning about an unrecognized field.** Mitigation:
  the spec restricts fields to `:param:`, `:returns:`, and `:raises:` — all
  recognized by MyST's fieldlist extension. If a different field slips in
  (e.g., `:see:`), rewrite as prose.

## Testing and verification

The change is verified when **all** of the following hold:

1. **Lint clean:**
   - `uv run ruff check` exits 0 (with `convention = "pep257"`).
   - `uv run mypy` exits 0 (unaffected by docstring style).

2. **Build clean:**
   - `./tools/build-docs.sh` exits 0 (`-W` Sphinx build).

3. **Content parity:**
   - `grep -rn 'Args:\|Returns:\|Yields:\|Raises:' mononet/ tools/`
     returns no matches (all converted).
   - The new `:param:`, `:returns:`, `:raises:` fields appear in source:
     `grep -rn ':param \|:returns:\|:raises ' mononet/ tools/` shows the
     converted blocks.

4. **Rendering quality:**
   - Open `docs/_build/html/apidocs/mononet.core.reference.html` in a
     browser. The `monotonic_dense` page must show parameters as a
     structured argument list (each parameter on its own row with name and
     description in distinct visual columns), not as a flat paragraph.
   - Same check for the `MonoLinear` page in
     `docs/_build/html/apidocs/mononet.torch.layers.html`.

5. **Tooling unaffected:**
   - `uv run pytest -q` still passes.
   - `uv run pre-commit run --all-files` still passes.

## Open questions

None blocking.

## Future work (deferred)

- Cross-referencing types in field descriptions (e.g., `:param mask: A
  {py:class}` `mononet.core.types.MonotonicityMask`` describing …`). Optional
  polish; the type annotation in the signature already cross-references via
  intersphinx-style links once mononet has its own API target indexed.
- A `docs/contributing.md` or `CONTRIBUTING.md` section showing the canonical
  docstring style with examples (currently the docs only mention the
  convention by name).
