# Design: Documentation audit + quick-win fixes

**Date:** 2026-07-12
**Status:** Approved (brainstorming) — pending implementation plan
**Scope:** Reader-facing documentation. Produces a prioritized audit report and fixes the quick-win findings in the same effort; larger findings become follow-up specs.

## Problem

The docs are already substantial (Sphinx + myst-nb: `index`, `installation`,
`concepts/`, `guides/`, `benchmarks/`, `examples/`, `about/`, an autodoc2 API
reference, plus README and paper references). But there is no systematic view of
where they fall short for the people who read them, and recent code changes
(default `mode` → `absolute`, default activation → `identity`, mandatory
activation on residual blocks) are prime staleness suspects. We want an
evidence-based, prioritized audit — and to fix the obviously-correct, low-effort
items immediately rather than filing them.

## Goals

- A single prioritized **audit report** covering every reader-facing surface.
- **Quick-win fixes** applied in the same pass (low effort + clearly correct).
- A ranked **follow-up list** so larger items can be picked off as their own
  specs.

## Non-goals

- No authored new pages or IA restructuring in this pass (those are follow-ups).
- Out of scope: `docs/superpowers/` internal specs/plans, the paper PDFs, and any
  code change beyond what a doc fix strictly requires.

## Audience & ranking

The audit weights findings by reader impact, in this order:

1. **New adopter / evaluator** and **Practitioner integrating** (highest — the
   largest audience for a published PyPI package).
2. **Researcher / reproducer**.
3. **Contributor**.

Every finding is tagged with the persona(s) it hurts; ties break toward the
higher-ranked persona.

## Surfaces audited

`docs/index`, `docs/installation`, `docs/concepts/*`, `docs/guides/*`
(pytorch/jax/keras), `docs/benchmarks/*`, `docs/examples/*`, `docs/about/*`, the
**autodoc2 API reference** (`docs/apidocs/`), and the repository **README**
(first thing a new adopter reads). The PyPI long-description is covered
transitively if it is derived from the README.

## Audit rubric

Each surface is scored on six dimensions. A finding names the surface, the
dimension, and the affected persona(s):

1. **Accuracy / staleness** — prose and snippets match current code. Prime
   suspects: default `mode` → `absolute` (#77), default activation → `identity`
   (#75), mandatory activation on residual blocks (#75). Snippets not already
   guarded by `tests/examples/` (README parity + per-backend `risk_net`) are a
   staleness risk and are checked by hand against the current API.
2. **Completeness / gaps** — a coherent path exists for each persona. Explicit
   checks: a true getting-started/quickstart (install → first monotonic model in
   ~10 lines) distinct from the deeper guides; migration/upgrade notes for the
   recent default changes.
3. **Structure / navigation** — toctree coverage (no orphan pages), landing-page
   routing to each persona, cross-linking among concepts ↔ guides ↔ API ↔
   benchmarks.
4. **API reference quality** — autodoc2 output is usable; public symbols have
   docstrings; docstrings conform to the MyST field-list spec
   ([2026-05-22-myst-docstrings-design.md](2026-05-22-myst-docstrings-design.md)).
5. **Build health** — add an internal+external **link check** and **nitpicky**
   cross-reference checking, and catalogue what they surface. (`sphinx-build -W`
   in `tools/build-docs.sh` already guards warnings.)
6. **Consistency / voice** — terminology and naming uniform across backends
   (`MonoLinear`/`MonoDense`, `MonoResidual`, `MonoInput`); tone matches the
   repo's terse senior-collaborator posture.

## Prioritization

Each finding carries:

- **Severity** — blocks/misleads a persona (high) → polish (low).
- **Effort** — quick win vs. follow-up.
- **Persona impact** — which reader(s), using the ranking above.

Findings are ordered most-severe-first, then highest-audience-first.

### Quick-win criteria (fixed in this pass)

Low effort **and** clearly correct — no design or authoring judgment required.
Examples: stale defaults / renamed APIs in prose, broken links or cross-refs the
new checks surface, missing cross-links, orphan pages, small factual/typo fixes,
docstring-format nits.

Explicitly **not** quick wins (→ follow-up specs): writing a new quickstart page,
restructuring information architecture, filling a conceptual gap, authoring a
migration guide — anything requiring new authored content or navigation
decisions.

### Explicit quick-win: README badges

The README currently carries PyPI-version, Python-versions, Docs, and Build
badges. Add, in this pass:

- **Codecov coverage** — `https://codecov.io/gh/davorrunje/mononet` graph badge,
  now meaningful with the 100% gate live. If the public badge renders blank, use
  the non-secret graph token from Codecov settings (this is the badge/graph
  token, **not** the upload `CODECOV_TOKEN`).
- **License** — Apache-2.0 (the repo emphasizes its license posture; see
  `NOTICE.md`).
- **arXiv** — links to arXiv:2205.11775 (the source paper), fitting for a
  paper-backed package.

Badges are placed in the existing badge block at the top of the README, ordered
so the most decision-relevant ones for a new adopter come first
(version/Python/license, then coverage/build/docs, then arXiv). A downloads or
code-style badge is intentionally omitted to avoid clutter.

## Deliverable

One audit report at `docs/superpowers/audits/2026-07-12-docs-audit.md`
containing:

1. **Ranked findings table** — columns: surface, dimension, persona(s),
   severity, effort, recommendation.
2. **Fixed in this pass** — the quick-win items, each linked to its commit.
3. **Recommended follow-ups** — larger findings, ordered so the next spec is
   obvious to pick.

Quick-win fixes land as focused commits on a branch and ship as one PR alongside
the report.

## Tooling added

- **Link check** — Sphinx `linkcheck` builder (internal + external URLs).
- **Nitpicky** cross-reference checking — surfaces dangling `:py:` / doc
  references.

Both are added so they *can* run locally and in CI, and are used to generate
build-health findings. **Wiring either into the CI merge gate is itself a
follow-up decision, not a quick win** — this pass only makes them runnable and
records what they find.

## Validation of the quick-win fixes

- `./tools/build-docs.sh` (strict `-W` build) stays green.
- The new `linkcheck` and nitpicky checks pass (or their remaining failures are
  explicitly catalogued as follow-ups, e.g. flaky external URLs).
- `uv run pytest tests/examples` still passes (guards README + `risk_net`
  parity).

## Verification

The report exists with all three sections populated; every quick-win finding is
either fixed (with a commit reference) or explicitly reclassified as a follow-up
with a one-line reason; the strict docs build and `tests/examples` are green on
the branch.
