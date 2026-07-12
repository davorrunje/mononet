# Documentation Audit — 2026-07-12

Audit of reader-facing docs against the 6-dimension rubric in
[the spec](../specs/2026-07-12-docs-audit-design.md). Personas ranked:
new-adopter ≈ practitioner > researcher > contributor.

Ground truth for API/defaults: `mononet/core/config.py`,
`mononet/{torch,jax,keras}/layers.py` (defaults: `mode="absolute"`,
`activation=identity`, `MonoResidual` activation mandatory). Build-health
evidence folded in from [`build-health-findings.txt`](build-health-findings.txt)
(Task 1). Dimensions: accuracy, completeness, structure, api-reference,
build-health, consistency. Rows are ordered most-severe first, then by
highest audience within a severity tier.

## Ranked findings

| # | Surface | Dimension | Persona(s) | Severity | Effort | Finding & recommendation |
|---|---------|-----------|-----------|----------|--------|--------------------------|
| 1 | `docs/index.md:7` | accuracy | new-adopter, practitioner | high | quick-win | Landing tagline reads "**Unconstrained** monotonic neural networks" — the package implements **Constrained** MNNs (README, paper title, `about/`). The flagship one-liner on the docs home is wrong. Change "Unconstrained" → "Constrained". |
| 2 | `docs/index.md` (landing) | structure | new-adopter, practitioner | med | follow-up | Landing page is only an install snippet + BibTeX + a hidden `toctree`; no "first monotonic model in ~10 lines" and no persona/"start here" routing. A new adopter lands with nowhere obvious to go next. README already has a runnable quick-start block — mirror a trimmed version (or `literalinclude` an `examples/` file) on the landing and add a short "New here? → Guides / Concepts" pointer. (Authored content → follow-up.) |
| 3 | `docs/` (no upgrade guide) | completeness | practitioner, researcher | med | follow-up | The recent breaking default changes (`mode`→`absolute`, `activation`→`identity`, mandatory `MonoResidual` activation) are documented only in the `about/changelog` "Changed" section. There is no adopter-facing upgrade/migration note in the guides or landing. Add a short "Upgrading" section (or callout in the guides) with the one-line recovery (`mode="switch"`, `activation="relu"`). Partially mitigated by the changelog. |
| 4 | `docs/apidocs/…/mononet.torch._kernels` (src `mononet/torch/_kernels.py:5`) | accuracy | researcher, contributor | med | quick-win | Module docstring says "Wrapper classes in layers.py / **models.py** …". `models.py` / composed-model classes were dropped from the project, so this names a file that never ships. myst-linkify also autolinks the bare filenames into two broken external URLs (`http://layers.py`, `http://models.py` — both fail linkcheck). Reword to reference only `layers.py`, wrapping it in backticks so linkify does not turn it into a URL. (jax/keras `_kernels.py` are single-line and clean.) |
| 5 | `docs/guides/pytorch.md` + `docs/apidocs/**` | build-health / api-reference | researcher, contributor | med | follow-up | Intersphinx cannot resolve many external targets: authored xrefs `torch.nn.Linear` / `torch.nn.Sequential` in the PyTorch guide report "Anchor not found", and the 350 nitpicky warnings are unresolved `numpy.typing.NDArray`/`DTypeLike`, `jax.numpy.ndarray`, `flax.nnx.*`, `keras.layers.Layer`, and inherited `torch.nn.Module` internals in the autodoc2 pages. Root causes: no flax/keras intersphinx inventory, `numpy.typing` unmapped, stale torch anchors, and autodoc2 emitting inherited base-class members. Fix via intersphinx config (add flax/keras inventories, refresh torch) + suppress inherited members / add `nitpick_ignore`. Config+design work → follow-up; the bulk is inherited-member noise, not per-symbol doc defects. |
| 6 | `docs/about/contributing.md` (vs `CONTRIBUTING.md`) | consistency / accuracy | contributor | med | follow-up | This page is a generic cookiecutter "Contributing to mononet **Documentation**" page scoped to doc edits only; it diverges from and omits the canonical dev workflow in the top-level `CONTRIBUTING.md` (five devcontainer flavors incl. `proofs`, `uv sync`, per-backend `pytest`, the ruff/mypy/bandit/semgrep pre-commit gate, release process, commit/PR + coding conventions). A contributor reaching it via the About nav misses the real workflow. Replace with an `{include}` of / pointer to `CONTRIBUTING.md` (mirroring how `about/changelog` mirrors `CHANGELOG.md`), or rewrite to match. Content rewrite/IA decision → follow-up. |
| 7 | `README.md:3-6` | completeness / consistency | new-adopter | low | quick-win | Badge row (PyPI version, Python versions, Docs, Build) has **no License badge**, despite Apache-2.0 being a first-class posture item (CLAUDE.md, `NOTICE.md`, `about/license`). Add a shields.io Apache-2.0 license badge linking to `LICENSE`. (Task 3.) |
| 8 | `about/changelog` (`CHANGELOG.md:82-83`) | build-health | contributor, researcher | low | quick-win | Footer link-refs `[Unreleased]: …/compare/v0.1.0...HEAD` and `[0.1.0]: …/releases/tag/v0.1.0` both 404 — v0.1.0 is not yet released. Point the compare link at `…/commits/main` (or `compare/main`) and drop/again-guard the `v0.1.0` tag ref until the first release. |
| 9 | `about/changelog` (`CHANGELOG.md:14,24,78`) | accuracy / consistency | researcher, contributor | low | quick-win | Stale entries in the "Added" block: `MonoLinearConfig` (renamed to `MonoConfig`), "stub layers raising `NotImplementedError`" (algorithm is now implemented), and "MkDocs site rewrite" / "Documentation framework with MkDocs Material" (project migrated to Sphinx). Correct the class name and the MkDocs→Sphinx references (clearly-correct string fixes); broader changelog curation can follow. |
| 10 | `apidocs/…/mononet.core.config`, `mononet.core.types` | api-reference | researcher | low | follow-up | Dataclass value objects document their fields in prose, not field lists: `MonoConfig` (6 fields), `MonotonicityMask`, `ActivationSpec`, `InitSpec` have no `:param:`/attribute entries, while `MonoResidualConfig` does use `:param activation:`. Inconsistent with the MyST field-list spec. Add `:param:` field lists to the dataclass docstrings. Authoring → follow-up. |
| 11 | `about/license` (`docs/about/license.md:13`) | build-health | researcher | low | follow-up | External link `https://patents.justia.com/patent/11551063` returns 403 in linkcheck — likely a bot block rather than a dead page (loads in a browser). Verify manually; if it is a checker-only 403, add it to `linkcheck_ignore` rather than removing the citation. |
| 12 | `docs/about/contributing.md:23` | build-health / consistency | contributor | low | quick-win | Uses `!!! note` (Material-for-MkDocs admonition syntax) left over from the MkDocs→Sphinx migration; MyST does not render it as an admonition — it appears as literal "!!! note" text followed by an indented block. Per [`2026-05-22-sphinx-migration-design.md`](../specs/2026-05-22-sphinx-migration-design.md) the correct form is a ```` ```{note} ```` fenced directive. Mechanical conversion → quick-win. |
| 13 | `docs/index.md` toctree → `docs/releasing.md` | structure | contributor | low | follow-up | The maintainer release runbook (`releasing.md`) sits in the top-level `index.md` toctree beside user-facing pages (installation, guides, concepts, benchmarks), mixing maintainer content into the primary nav. Move it under `about/` (next to `contributing`). IA change → follow-up. |

**Surfaces with no issues found:** `docs/installation.md` (extras table verified against `pyproject.toml`), `docs/concepts/*` (monotonicity, layers, monotonic-residual, proofs — defaults and gate math current), `docs/guides/jax.md` + `docs/guides/keras.md`, `docs/benchmarks/*` (index + notebooks/protocol — no stale defaults), `docs/examples/*` (`risk_net_{torch,jax,keras}.py`, surfaced via `literalinclude` in the guides and pinned by `tests/examples/`), `docs/about/index.md` (toctree-only landing for the About section — clean), and `README.md` prose/quick-start (accurate; only the badge gap in #7). Redirects flagged by linkcheck (pydata-sphinx-theme, sphinx-doc, PyPI `manage/account`) are benign upstream 301/303s, not findings.

## Fixed in this pass

_(completed in Tasks 3–4 — quick-win findings, each with its commit SHA on `spec/docs-audit`)_

- [x] #7 — README Apache-2.0 (+ codecov, arXiv) badges — `df161f0`
- [x] #1 — `docs/index.md:7` "Unconstrained" → "Constrained" — `fa2f6ba`
- [x] #4 — `mononet/torch/_kernels.py` drop `models.py`, backtick `layers.py` (also removes the broken `http://layers.py` / `http://models.py` linkify autolinks) — `fa2f6ba`
- [x] #12 — `docs/about/contributing.md` `!!! note` → ```` ```{note} ```` — `fa2f6ba`
- [x] #8 (partial) — `CHANGELOG.md` footer link-refs: `[Unreleased]` repointed from the nonexistent `compare/v0.1.0...HEAD` to `.../commits/main`. The `[0.1.0]` ref is left as the tag URL (`.../releases/tag/v0.1.0`) — it is a not-yet-created release tag, deferred to follow-ups rather than substituted with a different link (which would mislabel the 0.1.0 snapshot). — `fa2f6ba`, `1526b56`
- [x] #9 — `CHANGELOG.md` stale entries: `MonoLinearConfig` → `MonoConfig`, dropped the "stub layers raising `NotImplementedError`" line (algorithm is implemented), "MkDocs site rewrite" / "Documentation framework with MkDocs Material" → Sphinx (myst-nb) — `fa2f6ba`

Validation (all green on `spec/docs-audit` after the fixes above):
- `./tools/build-docs.sh` → `build succeeded`.
- `./tools/check-docs.sh` → the `http://layers.py` / `http://models.py` linkify breaks are gone; all remaining `linkcheck` failures are external (pytorch-docs anchors #5, `patents.justia.com` 403 #11) — no internal link/xref regressions.
- `uv run pytest tests/examples -q` → `4 passed`.

## Recommended follow-ups

_(every open finding, ordered by severity then audience — new-adopter/practitioner first — so the top row is the obvious next spec)_

1. **#2 Landing quickstart + start-here routing** (structure, new-adopter/practitioner, med) — a short authored quickstart on `docs/index.md` (mirror the README block or `literalinclude`) plus persona pointers. Needs authored content/IA judgment → not a quick-win.
2. **#3 Upgrade/migration note** (completeness, practitioner/researcher, med) — small authored "Upgrading" section covering the three breaking default changes (`mode`, `activation`, mandatory `MonoResidual` activation); pairs naturally with #2. Needs authored content → not a quick-win.
3. **#5 Intersphinx / autodoc2 cleanup** (build-health/api-reference, researcher/contributor, med) — add flax/keras intersphinx inventories, map `numpy.typing`, refresh the torch inventory, suppress inherited base-class members, add targeted `nitpick_ignore`; clears the ~350-warning backlog and the `torch.nn.Linear`/`torch.nn.Sequential`/`torch.nn.Module`/`torch.dtype`/`torch.Tensor`/`torch.nn.parameter.Parameter` anchor breaks still present in `check-docs.sh` output together. Config-level design work → not a quick-win.
4. **#6 Reconcile `about/contributing.md` with `CONTRIBUTING.md`** (consistency/accuracy, contributor, med) — mirror/`{include}` the canonical dev-workflow doc (or rewrite the docs page to match) so contributors see the real five-devcontainer-flavor/`uv sync`/pre-commit-gate workflow. Content rewrite/IA decision → not a quick-win.
5. **#10 Field-list docstrings on core dataclasses** (api-reference, researcher, low) — bring `MonoConfig`/`types.py` in line with the MyST field-list spec and `MonoResidualConfig`. Authoring → not a quick-win.
6. **#11 `linkcheck_ignore` for the `patents.justia.com` 403** (build-health, researcher, low) — confirmed still 403ing under `check-docs.sh` in this pass (loads fine in a browser); after manual confirmation it's a bot block, add it to `linkcheck_ignore` rather than removing the citation.
7. **#8b CHANGELOG release-tag links** (build-health, contributor, low) — the `[0.1.0]` footer link-ref (`.../releases/tag/v0.1.0`) resolves once the first release is tagged; verify (or drop the ref) at release time. Deferred from #8's quick-win because substituting a live link now would mislabel the 0.1.0 snapshot.
8. **#13 Move `releasing.md` under `about/`** (structure, contributor, low) — one-page IA fix; can ride along with #6 since both touch the About section.
