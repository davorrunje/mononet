# honest-scholar integration — design

**Date:** 2026-07-21
**Status:** approved (brainstorming) — execution in progress
**Owner:** Davor Runje

## Purpose

Adopt the released [`honest-scholar`](https://github.com/davorrunje/honest-scholar)
plugin (v0.1.0, on PyPI, docs at <https://honest-scholar.science>) as the
research-workflow driver for `mononet`, and onboard this repository onto it with
the plugin's own `research-init` skill in **`adopt`** mode.

`honest-scholar` is the domain-neutral *science* layer (hypothesis → paper →
thesis, plus `literature`, `dataset`, `progress`, `defend`). `mononet` is one
**consumer** of it — in fact the reference `research-init adopt` target. This
document records how the two fit together and the concrete steps to wire them up.

## Background: current state

- **`honest-scholar` is released** (v0.1.0): 10 skills (`research-init`,
  `hypothesis-exploration`, `hypothesis-testing`, `paper-exploration`,
  `paper-synthesis`, `thesis`, `literature`, `dataset`, `progress`, `defend`) +
  a companion PyPI CLI. It governs by two principles — **agency** (humans make
  and sign every material decision; AI drafts) and **understanding** (`defend`
  is a Socratic examiner) — over a **git-native plain-text** layout.
- **`mononet` already runs the superpowers methodology** (brainstorming → spec →
  plan → implement) via `.claude/settings.json`. ~50 specs + ~40 plans live under
  `docs/superpowers/`.
- **In-repo design work now superseded by the released plugin:** PR #131
  (`design/scholar-plugin`, still points at the old `davorrunje/scholar`
  name — stale), PR #128 (research-workflow skill specs). PR #127
  (benchmark-experiment-orchestration) is **not** superseded — it is the
  experiment-backend *implementation* design and stays open (see §6).
- **Research assets `adopt` will map:**
  - *Literature:* `docs/references/` — Runje 2023 (arXiv:2205.11775, anchor) and
    Sartor 2025 (arXiv:2505.02537, rival/advancing), each with a curated digest.
  - *Datasets:* `benchmarks/datasets/` (manifest.toml + download/loader/registry/
    sources) covering `adult, german, lc, polish, taiwan`, plus config-driven
    `auto, blog, compas, heart, loan`. Some raw data is checked in (11M+).
  - *Experiment backend:* `benchmarks/` (`run.py`, `search.py`,
    `_common/results.py`) emitting result JSONs
    (`dataset/flavor/best_params/cv_best/test_*/n_seeds`) under
    `benchmarks/results/{phase2,alternate-base,deep-init,deep-residual,
    monoresidual-gate,size-ladder,...}`.

## Architecture: three layers

`honest-scholar` sits on top and **delegates outward** through its two contracts
rather than reimplementing engineering or experiments:

| Layer | Owned by | Binding |
|---|---|---|
| **Science** — hypotheses, papers, thesis, literature, datasets, defend, progress | `honest-scholar` plugin | n/a (the plugin) |
| **Engineering** — design, plan, implement, test | superpowers methodology | `engineering_backend: superpowers` |
| **Experiment** — run, evidence, tables, is-current | `benchmarks/` harness | `experiment_backend: benchmarks/` |

- The science skills hand code work off to the **engineering backend** via the
  engineering-delegation contract. `docs/superpowers/` stays exactly as-is and
  becomes the engineering record the science layer references.
- `findings.md` in each hypothesis cite `benchmarks/results/*.json` as the
  evidence produced by the **experiment backend**.

This keeps both existing systems intact: no rework of superpowers, no rework of
the benchmark harness — only new binding metadata in `.honest-scholar/config.yml`.

## Plan

### 1. Install the plugin (project-wide)

Edit `.claude/settings.json` (mirrors the existing superpowers install):

- add `honest-scholar` to `extraKnownMarketplaces`
  (`source: { source: github, repo: davorrunje/honest-scholar }`, `ref: v0.1.0`);
- enable `honest-scholar@honest-scholar` in `enabledPlugins`.

Install the CLI **isolated** from mononet's ML environment:

```bash
uv tool install honest-scholar
honest-scholar doctor
```

Do **not** add `honest-scholar` to `pyproject.toml` — it must not enter the ML
dependency tree. The `literature`, `dataset`, `defend`, and `progress` skills
call this CLI.

> **Session note:** enabling the plugin in `settings.json` does not hot-load its
> skills into the *current* session. For this onboarding session, `research-init`
> is executed by following its `SKILL.md` directly; the settings change makes the
> full skill set available to subsequent sessions and to collaborators.

### 2. Skill dedup

- Replace mononet's `.claude/skills/create-issue` with the plugin repo's
  canonical version, and add `create-pr` from the plugin repo.
- Update the `CLAUDE.md` references accordingly (the "Follow-ups become GitHub
  issues" section and the PR conventions section).

These are dev-workflow skills the plugin does **not** distribute to consumers, so
they remain in mononet's `.claude/skills/`.

### 3. `research-init adopt` — deepest run

Run the plugin's `research-init` in `adopt` mode. Scaffold the consumer layout
and backfill from existing assets, **with named human sign-off on every material
classification** (agency principle). This is a thesis-by-publication repo, so the
optional `thesis/` tree is included.

Scaffold + populate:

- `.honest-scholar/config.yml` bindings: `engineering_backend: superpowers`,
  `experiment_backend: benchmarks/`, literature anchors = the two papers, rclone
  remote deferred (placeholder + `.example`).
- **Literature** → `docs/research/literature/references.json` (CSL-JSON) +
  `triage.yml`: Runje 2023 = *anchor*, Sartor 2025 = *rival*.
- **Datasets** → `datasets.yml` from the existing `benchmarks/datasets/manifest.toml`,
  computing checksums and recording source/license/**tier**. *License and tier
  are human-confirmed per dataset.*
- **Papers** (human-confirmed split; a starting point, expected to evolve):
  1. Core multi-backend implementation + paper reproduction (sub-projects A/B/C).
  2. Monotone-constructions methods paper (mixed / alternate / split flavors +
     init + ablations — the current active thread).
  3. Structure-Preserving PINNs application (PR #116).
  4. Injective-monotonic & normalizing flows (sub-project D — future).
- **Retroactive hypotheses** — map existing result sets (`phase2`,
  `alternate-base`, `deep-init`, `deep-residual`, `monoresidual-gate`,
  `size-ladder`, hp-search) to `hypotheses/<date-slug>/findings.md` filled from
  the result JSONs. Presented as **batched review tables**; each retroactive
  verdict carries a named human sign-off + date.
- **Thesis tree** — `thesis/aims.md` (through-line: constrained monotonic
  networks — construction → benchmarking → applications), `thesis/kappa/`,
  `thesis/milestones.yml`, chapter ↔ paper map.
- Regenerate `docs/research/dashboard.md` via `progress` (never hand-edited).

`adopt` is idempotent and non-destructive; it leaves scaffolding staged for
review, and does not commit as part of the skill.

### 4. Commit + PR

Commit the settings/skill changes, the spec, and the generated `docs/research/`,
`datasets.yml`, and `.honest-scholar/` scaffolding on
`chore/honest-scholar-integration`; open a PR. Use honest-scholar's commit
trailers on the adopt-generated artifacts:

```
Generated-with: honest-scholar (https://github.com/davorrunje/honest-scholar)
HonestScholar-Skill: research-init
```

### 5. Supersede stale design PRs

- **Close #131** (`design/scholar-plugin`, stale `scholar` name) as superseded by
  this integration.
- **Close #128** (research-workflow skill specs) as superseded — the design now
  lives in the released plugin (`docs/design/`, `decisions/`).
- Add pointer comments on both before closing.

### 6. Deferred follow-ups (GitHub issues)

Create self-contained issues (per the `create-issue` skill) for:

- **Fold open PR branches into the methodology.** Map each in-flight branch/PR
  (`feat/alternate-construction`, `feat/phase4-flavor-ablation`,
  `feat/deep-residual-accuracy`, `feat/screen-results-clean`,
  `spec/applications-structure-preserving-pinns`,
  `feat/stage2-unified-depth-benchmark`, …) to a hypothesis under the right
  paper, so in-flight engineering re-enters through the `hypothesis-testing`
  front door.
- **Experiment-backend binding hardening / possible redesign.** PR #127
  (benchmark-experiment-orchestration) stays open; it is the experiment-backend
  *implementation* the honest-scholar contract binds to. Track a possible
  redesign of the benchmark backend against the contract.

## Non-goals

- Rewriting or migrating `docs/superpowers/` into honest-scholar's doc tree — the
  superpowers record stays; it *is* the engineering backend.
- Adding `honest-scholar` to `pyproject.toml` or otherwise into the ML env.
- Folding the open feature PRs this session (deferred — §6).

## Guardrails (from honest-scholar)

- Human confirms every material classification (dataset license/tier,
  result→hypothesis mapping, retroactive verdicts).
- Git-native plain text is the source of truth; no external tracker replaces it.
- Domain-neutral scaffolding; content stays owned by mononet.
- Idempotent, non-destructive; nothing overwritten without confirmation.
