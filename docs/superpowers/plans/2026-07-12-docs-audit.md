# Documentation Audit + Quick-Win Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a prioritized documentation audit report across all reader-facing surfaces, and fix the quick-win findings (including new README badges) in the same pass; larger findings become follow-up specs.

**Architecture:** This is investigative + editorial work, not code. "Tests" are the real doc gates: the strict Sphinx build (`sphinx-build -W`), a new on-demand link/nitpicky check, and `pytest tests/examples` (guards README + per-backend example parity). Task 1 adds build-health tooling and captures its output; Task 2 conducts the audit against a fixed 6-dimension rubric and writes the report; Tasks 3–4 apply quick-win fixes (README badges, then everything else) and finalize the report.

**Tech Stack:** Sphinx + myst-nb, `sphinx-build` builders (`html`, `linkcheck`), shields.io / Codecov badges, `uv`, pytest.

**Spec:** [docs/superpowers/specs/2026-07-12-docs-audit-design.md](../specs/2026-07-12-docs-audit-design.md)

## Global Constraints

- **Branch:** `spec/docs-audit` (already checked out). Never commit to `main`.
- **Commit signing is broken in this container** (points at an absent macOS SSH key) — always `git commit --no-gpg-sign`.
- **Report path:** `docs/superpowers/audits/2026-07-12-docs-audit.md` (new dir).
- **Audience ranking** (drives severity): new-adopter ≈ practitioner (highest) > researcher/reproducer > contributor. Tag every finding with the persona(s) it hurts.
- **Quick-win = low effort AND clearly correct, no authoring/design judgment.** Examples: stale defaults/renamed APIs in prose, broken links/xrefs, missing cross-links, orphan pages, typos, docstring-format nits, README badges. NOT quick wins: new pages (quickstart), IA restructuring, conceptual gaps, migration guides — those are follow-ups.
- **Do NOT wire linkcheck or nitpicky into the CI merge gate** (`.github/workflows/build.yml`) — this pass only makes them runnable and records findings. Wiring is a follow-up decision.
- **No code changes** beyond what a doc fix strictly requires (e.g. a docstring edit).
- **Surfaces in scope:** `docs/index`, `docs/installation`, `docs/concepts/*`, `docs/guides/*`, `docs/benchmarks/*`, `docs/examples/*`, `docs/about/*`, the autodoc2 API reference (`docs/apidocs/`), and `README.md`. Out of scope: `docs/superpowers/**`, paper PDFs.
- **Validation gates** (must stay green after any fix): `./tools/build-docs.sh`; `uv run pytest tests/examples`.
- **Facts:** `README.md` is NOT part of the Sphinx build (README edits can't break `-W`); `tests/examples/test_readme_matches.py` compares only ` ```python ` fences to `docs/examples/risk_net_*.py`, so badges (markdown image links) don't affect it. Hosted docs base URL: `https://davorrunje.github.io/mononet/`. License file: `LICENSE` (Apache-2.0). Paper: arXiv:2205.11775.

---

### Task 1: Build-health tooling (linkcheck + nitpicky) and captured findings

**Files:**
- Create: `tools/check-docs.sh`
- Create: `docs/superpowers/audits/build-health-findings.txt` (captured raw output; committed as audit evidence)

**Interfaces:**
- Produces: `tools/check-docs.sh` — runs a Sphinx `linkcheck` build and a nitpicky HTML build, printing broken links and dangling cross-references. Consumed by Task 2 (build-health dimension) and Task 4 (fixing broken links).

- [ ] **Step 1: Write `tools/check-docs.sh`**

```bash
#!/usr/bin/env bash
# On-demand documentation health checks: broken links (internal + external)
# and dangling cross-references (nitpicky). NOT wired into the CI gate — see
# docs/superpowers/specs/2026-07-12-docs-audit-design.md.
set -uo pipefail

echo "=== linkcheck (internal + external URLs) ==="
uv run sphinx-build -b linkcheck docs docs/_build/linkcheck
link_status=$?
echo "linkcheck exit: ${link_status}"
echo "(broken/redirected details: docs/_build/linkcheck/output.txt)"

echo
echo "=== nitpicky cross-reference check (warnings only, non-fatal) ==="
uv run sphinx-build -n -b html docs docs/_build/nitpick 2>&1 \
  | grep -iE "WARNING|ERROR" || echo "no nitpicky warnings"
```

- [ ] **Step 2: Make it executable and run it, capturing findings**

```bash
chmod +x tools/check-docs.sh
mkdir -p docs/superpowers/audits
./tools/check-docs.sh 2>&1 | tee docs/superpowers/audits/build-health-findings.txt
# Also copy the linkcheck detail file in, so the evidence is self-contained:
cp docs/_build/linkcheck/output.txt docs/superpowers/audits/linkcheck-output.txt 2>/dev/null || true
```

Expected: the script completes (linkcheck may report broken/redirected external URLs — that is data, not a failure of this task). `build-health-findings.txt` now contains the link + nitpicky results.

- [ ] **Step 3: Sanity-check the strict build still works untouched**

Run: `./tools/build-docs.sh`
Expected: `build succeeded` (this task added no `conf.py` changes, so the existing `-W` build is unaffected).

- [ ] **Step 4: Commit**

```bash
git add tools/check-docs.sh docs/superpowers/audits/build-health-findings.txt docs/superpowers/audits/linkcheck-output.txt
git commit --no-gpg-sign -m "docs(tooling): on-demand linkcheck + nitpicky check; capture build-health findings"
```

Note: `docs/_build/` is a build artifact directory — do NOT `git add` it. Only the two `docs/superpowers/audits/*.txt` evidence files and the script are committed.

---

### Task 2: Conduct the audit and write the report

**Files:**
- Create: `docs/superpowers/audits/2026-07-12-docs-audit.md`

**Interfaces:**
- Consumes: `docs/superpowers/audits/build-health-findings.txt` (Task 1).
- Produces: the audit report with a ranked findings table and a quick-win/follow-up classification for each finding. Task 3 (badges) and Task 4 (other quick wins) implement the quick-win findings and fill the report's "Fixed in this pass" section.

- [ ] **Step 1: Gather accuracy/staleness evidence against recent code changes**

Run these and record hits (file:line) as candidate findings. Recent code changes to verify docs against: default `mode` → `absolute`, default activation → `identity`, mandatory activation on `MonoResidual`.

```bash
# Any prose asserting an old default or a removed/renamed behavior:
grep -rnE "default.*(switch|relu)|mode *= *[\"']switch|convex_fraction|activation" \
  docs/index.md docs/installation.md docs/concepts docs/guides docs/benchmarks docs/about README.md
# Confirm current true defaults from the source of truth:
sed -n '1,60p' mononet/core/config.py   # MonoConfig/MonoResidualConfig defaults
grep -rn "activation" mononet/torch/layers.py | head
```

For every doc snippet NOT covered by `tests/examples/` (i.e. anything outside `docs/examples/risk_net_*.py` and the README python blocks), manually check it against the current public API in `mononet/{torch,jax,keras}/layers.py`. Record mismatches.

- [ ] **Step 2: Gather structure/navigation AND completeness/gaps evidence**

```bash
# Orphan pages (not in any toctree) surface as warnings from the strict build:
uv run sphinx-build -b html docs docs/_build/html 2>&1 | grep -iE "orphan|not in any toctree" || echo "no orphan warnings"
# Inventory toctrees to judge routing/coverage:
grep -rn "toctree" docs/index.md docs/*/index.md
# Completeness probes — is there a distinct getting-started/quickstart, and any
# migration/upgrade notes for the recent default changes?
grep -rniE "quick ?start|getting started|migrat|upgrad|changelog|breaking" docs README.md
```

Structure/navigation — judge: does the landing page (`docs/index.md`) route each persona (new-adopter, practitioner, researcher, contributor) somewhere sensible? Are concepts ↔ guides ↔ API ↔ benchmarks cross-linked? Record gaps.

Completeness/gaps — judge, per persona: is there a true getting-started/quickstart (install → first monotonic model in ~10 lines) distinct from the deeper per-backend guides? Are there migration/upgrade notes covering the recent default changes (`mode` → `absolute`, activation → `identity`, mandatory residual activation)? Record each missing path as a finding (these are typically `follow-up` effort, since they need authored content — but flag any that are genuinely a one-line cross-link as `quick-win`).

- [ ] **Step 3: Gather API-reference-quality evidence**

```bash
# Public symbols per backend (what the API reference should document):
grep -rnE "^class |^def |__all__" mononet/torch/layers.py mononet/jax/layers.py mononet/keras/layers.py mononet/core/*.py
```

Spot-check the generated API pages under `docs/apidocs/` and confirm public classes/functions have docstrings conforming to the MyST field-list spec (`docs/superpowers/specs/2026-05-22-myst-docstrings-design.md`: `:param x:`, `:returns:`, `:raises X:`, no `:type:`/`:rtype:`). Record missing docstrings or format violations.

- [ ] **Step 4: Fold in build-health findings and score consistency/voice**

Read `docs/superpowers/audits/build-health-findings.txt` (Task 1): each broken link, redirect, or dangling xref is a finding. Then scan for terminology/naming consistency (`MonoLinear`/`MonoDense`, `MonoResidual`, `MonoInput`) and tone across surfaces.

- [ ] **Step 5: Write the report**

Create `docs/superpowers/audits/2026-07-12-docs-audit.md` with exactly this structure. Fill the findings table from Steps 1–4; every finding gets surface, dimension (one of: accuracy, completeness, structure, api-reference, build-health, consistency), persona(s), severity (high/med/low), effort (quick-win/follow-up), and a concrete recommendation. Order rows most-severe-first, then highest-audience-first.

```markdown
# Documentation Audit — 2026-07-12

Audit of reader-facing docs against the 6-dimension rubric in
[the spec](../specs/2026-07-12-docs-audit-design.md). Personas ranked:
new-adopter ≈ practitioner > researcher > contributor.

## Ranked findings

| # | Surface | Dimension | Persona(s) | Severity | Effort | Finding & recommendation |
|---|---------|-----------|-----------|----------|--------|--------------------------|
| 1 | ... | ... | ... | ... | ... | ... |

## Fixed in this pass

_(completed in Tasks 3–4 — quick-win findings, each with its commit SHA)_

## Recommended follow-ups

_(follow-up findings, ordered so the next spec is obvious to pick)_
```

- [ ] **Step 6: Self-check coverage, then commit**

Confirm every in-scope surface (Global Constraints list) appears in at least one finding row OR is explicitly noted as "no issues found" in a closing line under the table. Confirm each finding has all columns filled and a quick-win/follow-up effort tag.

```bash
git add docs/superpowers/audits/2026-07-12-docs-audit.md
git commit --no-gpg-sign -m "docs(audit): prioritized documentation audit report (findings + classification)"
```

---

### Task 3: README badges (Codecov, License, arXiv)

**Files:**
- Modify: `README.md` (the badge block, lines 3–6)

**Interfaces:**
- Consumes: nothing. Produces: three new badges; recorded as a quick-win in the report's "Fixed in this pass" section in Task 4.

- [ ] **Step 1: Replace the badge block**

The current block (README.md lines 3–6) is:

```markdown
[![PyPI version](https://img.shields.io/pypi/v/mononet)](https://pypi.org/project/mononet/)
[![Python versions](https://img.shields.io/pypi/pyversions/mononet)](https://pypi.org/project/mononet/)
[![Docs](https://img.shields.io/badge/docs-mononet-blue)](https://davorrunje.github.io/mononet/)
[![Build](https://github.com/davorrunje/mononet/actions/workflows/build.yml/badge.svg)](https://github.com/davorrunje/mononet/actions/workflows/build.yml)
```

Replace it with this block (order per spec: version, Python, license, then coverage, build, docs, then arXiv):

```markdown
[![PyPI version](https://img.shields.io/pypi/v/mononet)](https://pypi.org/project/mononet/)
[![Python versions](https://img.shields.io/pypi/pyversions/mononet)](https://pypi.org/project/mononet/)
[![License](https://img.shields.io/pypi/l/mononet)](https://github.com/davorrunje/mononet/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/davorrunje/mononet/graph/badge.svg)](https://codecov.io/gh/davorrunje/mononet)
[![Build](https://github.com/davorrunje/mononet/actions/workflows/build.yml/badge.svg)](https://github.com/davorrunje/mononet/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-mononet-blue)](https://davorrunje.github.io/mononet/)
[![arXiv](https://img.shields.io/badge/arXiv-2205.11775-b31b1b.svg)](https://arxiv.org/abs/2205.11775)
```

- [ ] **Step 2: Verify README example parity is unaffected**

Run: `uv run pytest tests/examples/test_readme_matches.py -q`
Expected: PASS (badges are markdown image links, not ` ```python ` fences).

- [ ] **Step 3: Verify the badge URLs resolve**

```bash
for u in \
  "https://img.shields.io/pypi/l/mononet" \
  "https://codecov.io/gh/davorrunje/mononet/graph/badge.svg" \
  "https://img.shields.io/badge/arXiv-2205.11775-b31b1b.svg" ; do
  echo -n "$u -> "; curl -s -o /dev/null -w "%{http_code}\n" "$u"
done
```

Expected: each returns `200`. If the Codecov badge returns non-200 or renders blank, append the non-secret graph token from Codecov repo settings (`.../badge.svg?token=<graph-token>`) — this is the badge/graph token, NOT the upload `CODECOV_TOKEN`. Record in the report if the token was needed.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit --no-gpg-sign -m "docs(readme): add codecov, license, and arXiv badges"
```

---

### Task 4: Apply remaining quick-win fixes and finalize the report

**Files:**
- Modify: whichever in-scope doc files Task 2 flagged as quick-win fixes (e.g. `docs/concepts/*`, `docs/guides/*`, `docs/index.md`, `mononet/**` docstrings for format nits).
- Modify: `docs/superpowers/audits/2026-07-12-docs-audit.md` (fill "Fixed in this pass" and "Recommended follow-ups").

**Interfaces:**
- Consumes: the quick-win findings from Task 2's report; the badge fix from Task 3.

- [ ] **Step 1: Apply each quick-win finding**

For every row in the Task 2 report tagged `effort = quick-win` (excluding the badges done in Task 3), make the fix. Keep each logical group as its own edit. Typical fixes and how to make them:
- **Stale default / renamed API in prose:** edit the sentence/snippet to match the current API (verified against `mononet/core/config.py` and `mononet/*/layers.py`).
- **Broken internal link / dangling xref** (from `linkcheck-output.txt`): correct the target path or role.
- **Broken external URL:** update to the working URL; if the URL is dead with no replacement, remove the link and note it.
- **Missing cross-link / orphan page:** add the link or add the page to the appropriate `toctree`.
- **Docstring-format nit:** fix to the MyST field-list form (`:param:`/`:returns:`/`:raises:`), no `:type:`/`:rtype:`.

If any finding turns out to need authoring or design judgment, do NOT force it — reclassify it as a follow-up with a one-line reason (Step 3).

- [ ] **Step 2: Validate all gates**

```bash
./tools/build-docs.sh                      # strict -W build must succeed
./tools/check-docs.sh                      # re-run link/nitpicky; confirm fixed links resolve
uv run pytest tests/examples -q            # README + example parity intact
```
Expected: strict build `build succeeded`; previously-broken internal links/xrefs now clean (residual flaky external URLs are acceptable — catalogue them as follow-ups, don't block); `tests/examples` all pass.

- [ ] **Step 3: Finalize the report**

Edit `docs/superpowers/audits/2026-07-12-docs-audit.md`:
- Fill **"Fixed in this pass"** — list every quick-win applied (Task 3 badges + Task 4 fixes), each with its commit SHA (`git log --oneline` on this branch).
- Fill **"Recommended follow-ups"** — every `follow-up` finding plus anything reclassified in Step 1, ordered by (severity, audience) so the top row is the obvious next spec.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit --no-gpg-sign -m "docs: apply quick-win audit fixes and finalize audit report"
```

---

## After all tasks

Do NOT open the PR from within a task — the finishing step (superpowers:finishing-a-development-branch, run after the whole-branch review) handles push/PR. The PR bundles the report, the badges, the tooling, and the quick-win fixes; its description should link the audit report and summarize the recommended follow-ups so they can become their own specs.

## Notes for the implementer

- `docs/_build/` is a build artifact — never `git add` it. Only commit the `docs/superpowers/audits/*.md` / `*.txt` evidence + report, the `tools/check-docs.sh` script, and the doc/badge fixes.
- Never run any `uv sync` variant (it prunes backend extras in this container). The environment already has all backends installed.
- The audit is judgment work: when unsure whether something is a real problem for a persona, record it as a low-severity finding with your reasoning rather than silently dropping it — the report is meant to be complete, and the human triages.
