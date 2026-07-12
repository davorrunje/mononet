# Documentation Audit — 2026-07-12

Audit of reader-facing docs against the 6-dimension rubric in
[the spec](../specs/2026-07-12-docs-audit-design.md). Personas ranked:
new-adopter ≈ practitioner > researcher > contributor.

Ground truth for API/defaults: `mononet/core/config.py`,
`mononet/{torch,jax,keras}/layers.py` (defaults: `mode="absolute"`,
`activation=identity`, `MonoResidual` activation mandatory). Build-health
evidence folded in from [`build-health-findings.txt`](build-health-findings.txt)
(Task 1). Dimensions: accuracy, completeness, structure, api-reference,
build-health, consistency.

## Ranked findings

| # | Surface | Dimension | Persona(s) | Severity | Effort | Finding & recommendation |
|---|---------|-----------|-----------|----------|--------|--------------------------|
| 1 | `docs/index.md:7` | accuracy | new-adopter, practitioner | high | quick-win | Landing tagline reads "**Unconstrained** monotonic neural networks" — the package implements **Constrained** MNNs (README, paper title, `about/`). The flagship one-liner on the docs home is wrong. Change "Unconstrained" → "Constrained". |
| 2 | `apidocs/…/mononet.torch._kernels` (src `mononet/torch/_kernels.py:5`) | accuracy | researcher, contributor | med | quick-win | Module docstring says "Wrapper classes in layers.py / **models.py** …". `models.py` / composed-model classes were dropped from the project, so this names a file that never ships. myst-linkify also autolinks the bare filenames into two broken external URLs (`http://layers.py`, `http://models.py` — both fail linkcheck). Reword to reference only `layers.py`, wrapping it in backticks so linkify does not turn it into a URL. (jax/keras `_kernels.py` are single-line and clean.) |
| 3 | `docs/index.md` (landing) | structure | new-adopter, practitioner | med | follow-up | Landing page is only an install snippet + BibTeX + a hidden `toctree`; no "first monotonic model in ~10 lines" and no persona/"start here" routing. A new adopter lands with nowhere obvious to go next. README already has a runnable quick-start block — mirror a trimmed version (or `literalinclude` an `examples/` file) on the landing and add a short "New here? → Guides / Concepts" pointer. (Authored content → follow-up.) |
| 4 | `docs/` (no upgrade guide) | completeness | practitioner, researcher | med | follow-up | The recent breaking default changes (`mode`→`absolute`, `activation`→`identity`, mandatory `MonoResidual` activation) are documented only in the `about/changelog` "Changed" section. There is no adopter-facing upgrade/migration note in the guides or landing. Add a short "Upgrading" section (or callout in the guides) with the one-line recovery (`mode="switch"`, `activation="relu"`). Partially mitigated by the changelog. |
| 5 | `docs/guides/pytorch.md` + `docs/apidocs/**` | build-health / api-reference | researcher, contributor | med | follow-up | Intersphinx cannot resolve many external targets: authored xrefs `torch.nn.Linear` / `torch.nn.Sequential` in the PyTorch guide report "Anchor not found", and the 350 nitpicky warnings are unresolved `numpy.typing.NDArray`/`DTypeLike`, `jax.numpy.ndarray`, `flax.nnx.*`, `keras.layers.Layer`, and inherited `torch.nn.Module` internals in the autodoc2 pages. Root causes: no flax/keras intersphinx inventory, `numpy.typing` unmapped, stale torch anchors, and autodoc2 emitting inherited base-class members. Fix via intersphinx config (add flax/keras inventories, refresh torch) + suppress inherited members / add `nitpick_ignore`. Config+design work → follow-up; the bulk is inherited-member noise, not per-symbol doc defects. |
| 6 | `README.md:3-6` | completeness / consistency | new-adopter | low | quick-win | Badge row (PyPI version, Python versions, Docs, Build) has **no License badge**, despite Apache-2.0 being a first-class posture item (CLAUDE.md, `NOTICE.md`, `about/license`). Add a shields.io Apache-2.0 license badge linking to `LICENSE`. (Task 3.) |
| 7 | `about/changelog` (`CHANGELOG.md:82-83`) | build-health | contributor, researcher | low | quick-win | Footer link-refs `[Unreleased]: …/compare/v0.1.0...HEAD` and `[0.1.0]: …/releases/tag/v0.1.0` both 404 — v0.1.0 is not yet released. Point the compare link at `…/commits/main` (or `compare/main`) and drop/again-guard the `v0.1.0` tag ref until the first release. |
| 8 | `about/changelog` (`CHANGELOG.md:14,24,78`) | accuracy / consistency | researcher, contributor | low | quick-win | Stale entries in the "Added" block: `MonoLinearConfig` (renamed to `MonoConfig`), "stub layers raising `NotImplementedError`" (algorithm is now implemented), and "MkDocs site rewrite" / "Documentation framework with MkDocs Material" (project migrated to Sphinx). Correct the class name and the MkDocs→Sphinx references (clearly-correct string fixes); broader changelog curation can follow. |
| 9 | `docs/index.md` toctree → `docs/releasing.md` | structure | contributor | low | follow-up | The maintainer release runbook (`releasing.md`) sits in the top-level `index.md` toctree beside user-facing pages (installation, guides, concepts, benchmarks), mixing maintainer content into the primary nav. Move it under `about/` (next to `contributing`). IA change → follow-up. |
| 10 | `apidocs/…/mononet.core.config`, `mononet.core.types` | api-reference | researcher | low | follow-up | Dataclass value objects document their fields in prose, not field lists: `MonoConfig` (6 fields), `MonotonicityMask`, `ActivationSpec`, `InitSpec` have no `:param:`/attribute entries, while `MonoResidualConfig` does use `:param activation:`. Inconsistent with the MyST field-list spec. Add `:param:` field lists to the dataclass docstrings. Authoring → follow-up. |
| 11 | `about/license` (`docs/about/license.md:13`) | build-health | researcher | low | follow-up | External link `https://patents.justia.com/patent/11551063` returns 403 in linkcheck — likely a bot block rather than a dead page (loads in a browser). Verify manually; if it is a checker-only 403, add it to `linkcheck_ignore` rather than removing the citation. |

**Surfaces with no issues found:** `docs/installation.md` (extras table verified against `pyproject.toml`), `docs/concepts/*` (monotonicity, layers, monotonic-residual, proofs — defaults and gate math current), `docs/guides/jax.md` + `docs/guides/keras.md`, `docs/benchmarks/*` (index + notebooks/protocol — no stale defaults), `docs/examples/*` (`risk_net_{torch,jax,keras}.py`, surfaced via `literalinclude` in the guides and pinned by `tests/examples/`), and `README.md` prose/quick-start (accurate; only the badge gap in #6). Redirects flagged by linkcheck (pydata-sphinx-theme, sphinx-doc, PyPI `manage/account`) are benign upstream 301/303s, not findings.

## Fixed in this pass

_(completed in Tasks 3–4 — quick-win findings, each with its commit SHA)_

- [ ] #1 — `docs/index.md` "Unconstrained" → "Constrained"
- [ ] #2 — `mononet/torch/_kernels.py` drop `models.py`, backtick `layers.py`
- [ ] #6 — README Apache-2.0 license badge (Task 3)
- [ ] #7 — `CHANGELOG.md` footer link-refs (compare/tag)
- [ ] #8 — `CHANGELOG.md` stale `MonoLinearConfig` / MkDocs references

## Recommended follow-ups

_(follow-up findings, ordered so the next spec is obvious to pick)_

1. **#3 Landing quickstart + start-here routing** — highest-audience gap; a short authored quickstart on `docs/index.md` (mirror the README block or `literalinclude`) plus persona pointers.
2. **#4 Upgrade/migration note** — small authored "Upgrading" section covering the three breaking default changes; pairs naturally with #3.
3. **#5 Intersphinx / autodoc2 cleanup** — config-level: add flax/keras inventories, map `numpy.typing`, refresh torch, suppress inherited base-class members, add targeted `nitpick_ignore`; clears the 350-warning backlog and the guide anchor breaks together.
4. **#9 Move `releasing.md` under `about/`** — one-page IA fix.
5. **#10 Field-list docstrings on core dataclasses** — bring `MonoConfig`/`types.py` in line with the MyST spec and `MonoResidualConfig`.
6. **#11 `linkcheck_ignore` for the patents.justia 403** — after manual confirmation it is a bot block.
