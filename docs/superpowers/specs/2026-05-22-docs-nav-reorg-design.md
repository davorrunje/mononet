# Docs Navigation Reorganization — Design

**Date:** 2026-05-22
**Status:** Approved
**Author:** Davor Runje (with Claude Code)

## Problem

The current top-level navigation on the Sphinx docs site exposes too many
items. The home page `index.md` declares five captioned toctrees, but one of
them (`Getting started`) lists three framework guides — pytorch, jax, keras —
as siblings, so PyData Sphinx Theme renders them as separate navbar/sidebar
entries instead of one collapsible group. The result is a flat, crowded top
bar.

Additionally:

- `Citation` lives under `About` but is content users typically want near the
  landing page — the BibTeX block belongs on the home page.
- `API reference` is a verbose label; `Reference` is the conventional Sphinx
  term and reads better in the navbar.

## Goal

Reorganize so the top navigation shows **exactly five items**: **Guides ·
Concepts · Benchmarks · Reference · About**. Framework guides collapse under
**Guides**. The citation BibTeX moves to the home page.

## Non-goals

- Restyling landing pages beyond title + one-line intro + subtoctree.
- Adding redirects for the deleted `about/citation.html` path. (No internal
  links reference it; external 404s are acceptable for an alpha-stage site.)
- Reworking the autodoc2-generated API tree.
- Sidebar/footer/search customization.

## Architecture

A single **hidden** root toctree on `docs/index.md` lists five entries. Each
entry is a section *landing page* that owns its own subtoctree. PyData Sphinx
Theme renders one navbar item per root-toctree entry, so the top bar shows the
five intended items, each clickable to a landing page that organizes the
section's children in its sidebar.

```
docs/index.md  (hidden root toctree)
├── guides/index.md       → Guides       (subtoctree: pytorch, jax, keras)
├── concepts/index.md     → Concepts     (subtoctree: monotonicity, layers)
├── benchmarks/index.md   → Benchmarks   (existing page)
├── reference.md          → Reference    (wrapper; subtoctree: apidocs/mononet/mononet)
└── about/index.md        → About        (subtoctree: license, changelog, contributing)
```

## Files

### Create

**`docs/guides/index.md`**

```markdown
# Guides

Framework-specific quickstarts for mononet.

\`\`\`{toctree}
:maxdepth: 1

pytorch
jax
keras
\`\`\`
```

**`docs/concepts/index.md`**

```markdown
# Concepts

Background reading on monotonic neural networks.

\`\`\`{toctree}
:maxdepth: 1

monotonicity
layers
\`\`\`
```

**`docs/about/index.md`**

```markdown
# About

\`\`\`{toctree}
:maxdepth: 1

license
changelog
contributing
\`\`\`
```

**`docs/reference.md`**

```markdown
# Reference

API reference for the `mononet` package.

\`\`\`{toctree}
:maxdepth: 2

apidocs/mononet/mononet
\`\`\`
```

### Modify

**`docs/index.md`** — replace the five captioned toctrees with:

1. A new `## Citation` section containing the BibTeX block previously in
   `about/citation.md` (verbatim, including the "confirm the exact BibTeX
   entry against the PMLR proceedings page" note).
2. A single **hidden** root toctree:

   ```markdown
   \`\`\`{toctree}
   :hidden:

   guides/index
   concepts/index
   benchmarks/index
   reference
   about/index
   \`\`\`
   ```

The existing `# mononet`, abstract, and `## Install` sections stay as-is and
come above the Citation section.

### Move

**`docs/contributing.md`** → **`docs/about/contributing.md`**

The file content is unchanged. `about/index.md`'s subtoctree references it as
`contributing` (relative).

### Delete

**`docs/about/citation.md`** — content lives on the home page now.

## Navbar labels

PyData Sphinx Theme derives navbar item labels from the **page title** of each
root-toctree entry. With the wrapper for Reference, all five labels are
controlled by us:

- `guides/index.md` → "Guides"
- `concepts/index.md` → "Concepts"
- `benchmarks/index.md` → `# H1` of that page (currently "Benchmarks")
- `reference.md` → "Reference"
- `about/index.md` → "About"

## Testing

**Build gate:** `./tools/build-docs.sh` runs `sphinx-build -W` with warnings
treated as errors. Passing means:

- No broken `:doc:` references.
- No orphan pages (every page reachable via toctree).
- No duplicate toctree entries.

**Visual smoke test:** Run `./tools/serve-docs.sh` and confirm:

1. Top navbar shows exactly five items: Guides, Concepts, Benchmarks,
   Reference, About.
2. Clicking each opens a landing page whose sidebar lists its children.
3. The home page shows the Citation BibTeX section.
4. The previously broken `about/citation.html` URL is gone (acceptable).

## Risks

- **Sphinx caches stale outputs.** If a previous build cached
  `about/citation.html` or the old `contributing.html` location, a fresh
  build in CI starts from scratch and won't be affected. Local devs should
  `rm -rf docs/_build` if confused.
- **External inbound links** to `about/citation.html` will 404. The repo
  is alpha-stage and the page has been live for days, not years.

## Out of scope

- A versions switcher (already shipped via sphinx-polyversion).
- Search index tuning.
- Reordering or renaming child pages within each section.
- Adding new content to landing pages beyond a one-line intro.
