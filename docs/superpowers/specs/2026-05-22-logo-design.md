# mononet Logo — Design

**Status:** Draft for review
**Date:** 2026-05-22
**Owner:** Davor Runje
**Triggering event:** The current `docs/_static/logo.png` is a third-party asset the project does not have rights to use. It must be replaced with a wholly-owned mark.

## Goal

Ship a wholly-owned visual identity for `mononet` consisting of:

- A vector mark that reads as **"monotonic"** at first glance.
- Light- and dark-theme variants that the PyData Sphinx Theme can pick up automatically.
- A PNG fallback and a favicon, derived deterministically from the SVG source.
- A brand-color update in `docs/_static/extra.css` so headings/links/buttons stay coherent with the new mark.

## Non-goals

- Wordmark file. The PyData theme renders "mononet" as text via `html_title` next to the mark; the logo file is mark-only.
- Social-card / Open-Graph image (1200×630).
- README banner image, social avatars, t-shirt artwork, animated variants.
- Alternate aspect ratios (horizontal / vertical lockups). One square mark is enough for now.
- Migrating to a design-token system or theme-aware CSS variables beyond what extra.css already does.
- Trademark filing.

## Background

`docs/_static/logo.png` is a 168×168 RGBA PNG carried over from the MkDocs era. It was kept through the [Sphinx migration](2026-05-22-sphinx-migration-design.md) but the project does not have rights to redistribute it, so deploying docs publicly on GitHub Pages makes replacement urgent.

The Sphinx config already reuses one image for both themes:

```python
"logo": {
    "image_light": "_static/logo.png",
    "image_dark": "_static/logo.png",
},
```

PyData Sphinx Theme supports separate paths for `image_light` and `image_dark`, which lets us give each theme its own asset without runtime CSS tricks.

`docs/_static/extra.css` currently defines two brand variables (`--pst-color-primary: #003257` navy, `--pst-color-link: #48A8D8` sky blue). These were chosen for the old logo and will be replaced as part of this change.

## Design overview

**Mark — "Curve + axis + fill" (option D from brainstorming).** A smooth, monotonically increasing Bézier curve set inside a thin two-line coordinate frame (left + bottom axes), with a soft tinted fill under the curve. Reads as a plot of a monotonic function — i.e. the library's whole premise rendered as a tiny chart.

**Palette — violet single hue (option C from brainstorming).** Two shades:

- Light theme: `#7c3aed` (violet-600) — full-strength stroke, 22% opacity fill, axes at 45% opacity.
- Dark theme: `#a78bfa` (violet-400) — full-strength stroke, 25% opacity fill, axes at 45% opacity.

Chosen because (a) it's distinctive in the Python ML space where most peers go blue/orange, (b) it carries on both light and dark backgrounds with one stroke color shift, and (c) it reads "warm and curve-like" rather than "data-center cold."

**Source of truth: hand-authored SVG.** Two files, ~20 lines each. The mark's geometry is fixed and small enough that scripting it (matplotlib / cairo Python) would add code and dependencies for no benefit. SVG is diffable, infinitely sharp at every render size, and trivially editable.

**Rasterization: a one-shot Python script.** `tools/render-logo.py` invokes `cairosvg` to render the light SVG to a 256 px PNG (fallback / legacy consumers) and a 32 px favicon. Run locally, outputs are committed. Not part of the docs CI build — the SVGs are the canonical assets.

## Visual specification (mark)

ViewBox: `0 0 64 64` (square, fits the existing 168×168 slot without aspect mismatches).

Elements, in z-order:

1. **Horizontal axis** — `M 8 52 L 56 52`, stroke `1.5`, color = palette accent, opacity `0.45`.
2. **Vertical axis** — `M 8 52 L 8 12`, stroke `1.5`, color = palette accent, opacity `0.45`.
3. **Fill under curve** — `M 8 52 C 22 52, 28 18, 56 12 L 56 52 Z`, fill = palette accent, opacity `0.22` (light) / `0.25` (dark), no stroke.
4. **Curve stroke** — `M 8 52 C 22 52, 28 18, 56 12`, fill `none`, stroke = palette accent, stroke-width `5`, `stroke-linecap="round"`.

The Bézier control points `(22, 52)` and `(28, 18)` produce a sigmoid-like rise — flat at the bottom-left, steep through the middle, levelling out at the top-right. The curve is monotonically increasing across `[8, 56]`.

## Files

**Created:**

- `docs/_static/logo-light.svg` — light-theme mark. Stroke/axis/fill all use `#7c3aed`. ViewBox `0 0 64 64`.
- `docs/_static/logo-dark.svg` — dark-theme mark. Same geometry; all colors swap to `#a78bfa`. Slightly higher fill opacity (`0.25` vs `0.22`) so the fill remains readable against the dark theme's near-black.
- `docs/_static/favicon.png` — 32 px PNG render of `logo-light.svg`. Browsers serve favicons from PNG fine; no need for a multi-resolution `.ico`.
- `tools/render-logo.py` — small (~30 line) `cairosvg`-based script. CLI: `uv run python tools/render-logo.py`. Reads `docs/_static/logo-{light,dark}.svg`, writes `docs/_static/logo.png` (256 px from light) and `docs/_static/favicon.png` (32 px from light). Pure stdlib + cairosvg. Idempotent.

**Modified:**

- `docs/_static/logo.png` — replaced with the 256 px PNG render of `logo-light.svg`. Kept (rather than removed) so any external link to this exact filename — e.g. older cached docs builds, third-party listings — continues to resolve to a valid mononet mark instead of a 404.
- `docs/conf.py` — change `html_theme_options["logo"]` to point `image_light` at `_static/logo-light.svg` and `image_dark` at `_static/logo-dark.svg`. Add `html_favicon = "_static/favicon.png"`.
- `docs/_static/extra.css` — swap `--pst-color-primary` to `#7c3aed` and `--pst-color-link` to `#a78bfa`. The dark-theme `html[data-theme="dark"]` block keeps `--pst-color-link: #a78bfa` (same value as light for now; the lighter shade reads as accent on both surfaces).
- `pyproject.toml` — add `cairosvg>=2.7` to the `[dependency-groups] docs` group so `tools/render-logo.py` runs reproducibly under `uv run`.

**Unchanged:** `docs/_static/extra.js`, all other docs files. No content edits.

**Deleted:** none.

## Rasterization tool — `tools/render-logo.py`

Contract:

- Inputs: `docs/_static/logo-light.svg`, `docs/_static/logo-dark.svg` (paths hard-coded relative to repo root).
- Outputs: `docs/_static/logo.png` (256×256, from light), `docs/_static/favicon.png` (32×32, from light).
- Dependencies: `cairosvg` only (added to docs group).
- Behavior: idempotent. Re-running with unchanged SVGs produces byte-identical PNGs.

This is a developer tool, not a CI step. It runs locally when the SVGs change. Outputs are committed so docs CI doesn't need cairo system libs.

## Testing & verification

- **SVG renders correctly** in Firefox / Chrome / Safari at 24, 32, 64, 128, 256 px. (Manual eyeball — open each SVG directly.)
- **Sphinx build still succeeds** with the new logo paths: `./tools/build-docs.sh` finishes with `build succeeded` and no broken-image warnings.
- **Both themes render the right mark.** Open `docs/_build/html/main/index.html` in a browser, toggle PyData theme's light/dark switch, confirm the navbar logo swaps from the violet-600 mark to the violet-400 mark.
- **Favicon shows up.** Browser tab icon is the violet curve, not the Sphinx default.
- **PNG fallback is wholly owned.** `tools/render-logo.py` produces `logo.png` from the project's own SVG — no third-party imagery anywhere in the pipeline.

## Out-of-scope items, for future reference

- A social-card / OG image is genuinely useful for GitHub social previews and link unfurls. Not blocking the deploy; deferred.
- A wordmark (the typeset "mononet" word as an asset) would be needed for README banners and merch. The theme renders text fine for docs, so deferred.

## Risks

- **`cairosvg` system dependencies.** cairo requires libcairo on the dev machine. On Debian/Ubuntu devcontainers (which this project's `.devcontainer` is built on) `libcairo2` is typically already present; if not, the script fails clearly with `OSError: no library called "cairo-2" was found`. Documented in the tool's docstring; affects only local PNG regeneration, never CI.
- **Brand-color swap touches more than the logo.** Updating `extra.css` propagates the violet to every link, button, and accent on the site. Acceptable — the goal is coherent identity, not isolated logo replacement — but worth flagging so the diff isn't surprising.
- **PNG fallback drift.** If someone edits an SVG and forgets to re-run `tools/render-logo.py`, `logo.png` falls out of sync. Mitigation: a tiny note in the tool's docstring + the rasterized PNGs go in the same PR as their source SVGs. Not worth a pre-commit hook for this size of asset.

## Acceptance criteria

- [ ] No file under `docs/_static/` derives from third-party imagery.
- [ ] `docs/conf.py` points `image_light` and `image_dark` at distinct SVG files.
- [ ] `docs/_static/favicon.png` exists and `conf.py` sets `html_favicon`.
- [ ] `docs/_static/extra.css` uses `#7c3aed` for `--pst-color-primary` and `#a78bfa` for `--pst-color-link`.
- [ ] `tools/render-logo.py` runs to completion and regenerates the PNGs deterministically.
- [ ] `./tools/build-docs.sh` completes without new warnings.
- [ ] The deployed site at `https://davorrunje.github.io/mononet/` shows the new mark in both themes and a favicon in the browser tab.
