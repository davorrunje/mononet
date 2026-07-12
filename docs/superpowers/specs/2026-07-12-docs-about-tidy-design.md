# Design: About-section tidy-up (contributing include, releasing move)

**Date:** 2026-07-12
**Status:** Approved (brainstorming; user delegated completion) — pending implementation plan
**Scope:** Reader-facing docs `about/` section + top-level `CONTRIBUTING.md` links. Workstream C (final) of the documentation-audit follow-ups.

## Problem

Three findings from the [documentation audit](../audits/2026-07-12-docs-audit.md):

- **#4 (contributing divergence):** `docs/about/contributing.md` is a thin,
  generic "contribute to the documentation" page that diverges from and omits
  the canonical top-level `CONTRIBUTING.md` (license-of-contributions, five
  devcontainer flavors, uv, Claude Code provisioning, testing/lint/release/PR
  conventions). Doc-site contributors never see the real dev workflow.
- **#8 (releasing IA):** the maintainer runbook `docs/releasing.md` sits in the
  top-level `docs/index.md` toctree beside user-facing pages (installation,
  guides, concepts, benchmarks), mixing maintainer content into the primary nav.
- **#7 (CHANGELOG tag links):** deferred from the audit's quick-win pass — the
  CHANGELOG release-tag footer link resolves only once a release is tagged;
  fold a verification line into the release runbook since this workstream is
  already touching it.

## Decisions

- **#4 → `{include}` the canonical file.** `docs/about/contributing.md` becomes a
  thin wrapper that `{include}`s `../../CONTRIBUTING.md`, exactly mirroring how
  `docs/about/changelog.md` already includes `CHANGELOG.md`. Single source of
  truth, content renders on the docs site, zero drift.
- **#8 → move under `about/`.** `docs/releasing.md` → `docs/about/releasing.md`;
  re-point both toctrees.
- **#7 → one runbook line** in the moved `releasing.md`.

## Non-goals

- No rewrite/duplication of contributor content (that is what #4 is fixing).
- No CI-workflow change; no `mononet/**` code change.
- The deferred #3 upgrade note (first-release work) is still out of scope.

## Files

- Modify: `docs/about/contributing.md` — replace the generic page with a
  wrapper that `{include}`s `../../CONTRIBUTING.md`.
- Modify: `CONTRIBUTING.md` — absolutize its repo-relative file links so they
  resolve when included into the docs under the strict `-W`/nitpicky build.
- Move: `docs/releasing.md` → `docs/about/releasing.md` (`git mv`).
- Modify: `docs/index.md` — remove `releasing` from the top-level toctree.
- Modify: `docs/about/index.md` — add `releasing` to the About toctree.
- Modify: `docs/about/releasing.md` (post-move) — add the #7 CHANGELOG-tag-link
  verification line.

## #4 — contributing page as an include

Replace the entire body of `docs/about/contributing.md` with the changelog
pattern:

```markdown
# Contributing

The repository's `CONTRIBUTING.md` is authoritative — this page mirrors it.

```{include} ../../CONTRIBUTING.md
:start-line: 2
```
```

`:start-line:` is tuned (as in `changelog.md`) so `CONTRIBUTING.md`'s own
`# Contributing to mononet` H1 is not duplicated under the page's `# Contributing`
heading; the implementer confirms the exact value against the rendered build.

### CONTRIBUTING.md link absolutization

`{include}` does not rewrite relative links, and the strict `-W` build treats an
unresolved local-file link as fatal. So `CONTRIBUTING.md`'s six repo-relative
**file** links are rewritten to absolute GitHub blob URLs
(`https://github.com/davorrunje/mononet/blob/main/<path>`), which resolve both
on GitHub and from the included docs page:

- `NOTICE.md`, `SECURITY.md`, `PULL_REQUEST_GUIDE.md`
- `.devcontainer/claude-plugins.txt`
- `docs/superpowers/specs/2026-05-22-myst-docstrings-design.md`
- `docs/releasing.md` (both occurrences) → the **new** path
  `docs/about/releasing.md` (see #8)

The two intra-document anchor links (`#claude-code-plugins--sessions`,
`#lint-format-static-analysis`) are left as-is: `myst_heading_anchors` produces
GitHub-style slugs, so they resolve in the included page; the strict build
confirms this (if a slug mismatches, the implementer adjusts the link to the
myst slug — the headings themselves are unchanged).

Absolute `https://` links and anchors already present are untouched.

## #8 — move releasing.md under about/

- `git mv docs/releasing.md docs/about/releasing.md`.
- `docs/index.md`: remove the `releasing` entry from the top-level `{toctree}`.
- `docs/about/index.md`: add `releasing` to the About `{toctree}`, giving order
  `license`, `changelog`, `contributing`, `releasing`.
- The only inbound reference to `releasing` is the top-level toctree entry (plus
  the two `CONTRIBUTING.md` links handled above); README has none. The moved
  page's own content (absolute PyPI URLs) needs no path fixes.

## #7 — CHANGELOG tag-link verification in the runbook

Add one line to the moved `docs/about/releasing.md`, in the release steps near
the GitHub-Release/tag step:

> After the GitHub Release is published, confirm the CHANGELOG version/compare
> footer links resolve — the `v<version>` release-tag link only goes live once
> the release/tag exists.

## Validation

- `./tools/build-docs.sh` (strict `-W` **and** `nitpicky=True`, already enforced)
  exits 0, zero warnings — this catches: a broken `{include}`, unresolved
  intra-page anchors, dangling/orphan pages from the `releasing.md` move, and any
  dead local link left in the included `CONTRIBUTING.md`.
- `./tools/check-docs.sh` — linkcheck introduces no *new* internal breaks. Note:
  the newly-absolutized `blob/main/docs/about/releasing.md` link will 404 under
  linkcheck until this PR merges (the path exists only on the branch); this is an
  expected, self-healing transient and linkcheck is advisory (not a CI gate),
  matching how the CHANGELOG `v0.1.0` tag link is already handled.
- `uv run pytest tests/examples` — sanity that no doc-example wiring broke (docs
  content only; expected to pass unchanged).

## Verification

`about/` renders License → Changelog → Contributing (the full canonical dev
workflow) → Releasing; the top-level nav no longer lists `releasing`; the
`releasing.md` runbook carries the CHANGELOG-tag verification line; and the
strict build is green at zero warnings.
