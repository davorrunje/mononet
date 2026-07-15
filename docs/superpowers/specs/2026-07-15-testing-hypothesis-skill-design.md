# Testing-hypothesis skill

**Status:** design
**Date:** 2026-07-15
**Author:** Davor Runje
**Scope:** a new Claude Code skill + a standardized hypothesis-record format and folder
convention. Repo tooling/process — no impact on the `mononet` wheel.

## Problem

This is a research project: most experiments are attempts to verify a hypothesis. Today a
hypothesis lives informally — in agent memory (e.g. the depth-null thesis), scattered
across spec "Follow-ups", PR descriptions, and commit messages. There is no durable,
uniform record tying a hypothesis to the experiments that tested it and the verdict that
resulted. That record would be valuable twice over: as the backbone of the eventual paper,
and as a reproducible history of the research itself.

We want a **systematic, skill-driven way to record hypotheses and their tests**, with a
standardized document set and format, sufficient for reproducibility.

## Relationship to the orchestration spec

Sibling of [2026-07-15-benchmark-experiment-orchestration-design.md](2026-07-15-benchmark-experiment-orchestration-design.md).
That spec makes *experiments* reproducible (declarative specs, provenance, committed
results, run records). This spec makes the *hypotheses those experiments test* a durable
record, and leans on the orchestration provenance chain to be reproducible end-to-end. The
two are independent deliverables designed to interlock.

## Design principles

Three principles shaped the design:

1. **The record must not depend on things that only exist later.** At creation you have
   the *claim*, not the test — how to test it is produced by later stages. Paper layout is
   later still and emerges during write-up, so the record carries no paper-structure
   field; the eventual paper references hypotheses (by a stable handle), not the reverse.
2. **Science before engineering.** *How would we prove or refute this?* — the null, what
   evidence counts, confounds, the decision rule — is hard conceptual work that must
   precede any engineering. The engineering design (how to build/run the benchmarks)
   serves a scientific strategy that already exists. This is why `strategy.md` precedes
   `design.md`, and why the skill **owns the scientific reasoning** but **delegates the
   engineering** to `superpowers:brainstorming`.
3. **Structured, but not time-gated.** Required structure (a decision rule, a verdict
   backed by provenance) without enforcing a frozen before/after split. Realized by
   splitting the record into stage-documents created at their natural times — an
   *organizational* separation that makes the timeline legible without hard-gating edits.

## Layout

### The skill

Mirrors the existing `create-issue` skill shape:

```
.claude/skills/testing-hypothesis/
├── SKILL.md        # frontmatter + "When to use" + numbered, staged "Process"
└── TEMPLATE.md     # the standardized document templates (as create-issue has STYLE.md)
```

### Hypothesis folders — a parallel research track

Research (hypothesis-driven) work nests under a per-hypothesis folder; infra /
non-hypothesis work keeps the flat `docs/superpowers/specs/` + `plans/` convention. Two
tracks, deliberately. (This skill is itself infra, so *its* spec lives in the flat dir —
this file.)

```
docs/research/hypotheses/2026-07-15-depth-null/
├── hypothesis.md   # free-form claim + tiny frontmatter anchor      — created at CAPTURE
├── strategy.md     # scientific proof/refutation strategy           — created at STRATEGIZE
├── design.md       # engineering: how to build/run the experiments  — created at DESIGN (brainstorming)
├── plan.md         # writing-plans implementation checklist          — created at DESIGN
└── findings.md     # results + verdict + provenance                 — created at RECORD
```

Each document is created when its stage arrives; a hypothesis logged today has only
`hypothesis.md`. Folder id is **date-prefixed** `YYYY-MM-DD-<slug>`, consistent with the
repo's spec/plan filename convention.

### Index

`docs/research/hypotheses/README.md` — a registry table (`status · label · slug ·
one-line claim · link`), the research analogue of `MEMORY.md`. The skill appends a row on
create and rewrites it on verdict. Optionally regenerable by scanning all `hypothesis.md`
frontmatter, so it cannot silently drift; a maintained table is acceptable for v1.

## Documents

### `hypothesis.md` — the anchor (free-form + frontmatter)

A small machine-readable YAML header for the index, then a free-form prose body — near-zero
ceremony to bank an idea.

```yaml
---
title: Depth does not help constrained monotone networks
slug: depth-null
label: H-depth        # optional stable citation handle for the eventual paper
created: 2026-07-15
status: open          # open | supported | refuted | inconclusive | superseded
authors: [Davor Runje]
tags: [depth, expressivity, monotonic]
related: [monoresidual-gate-instability]
verdict: null         # {outcome, date, run_hashes:[...]} — set at RECORD
---
```

Body: the claim in prose, the intuition, why it might be true, prior work. Free-form by
design — the goal is to capture the idea, not to structure it prematurely.

### `strategy.md` — the science (how to prove/refute)

The skill's distinctive artifact, authored through a guided scientific-reasoning dialogue
*before* any engineering. Its required sections encode the rigor practices (see
[Scientific rigor](#scientific-rigor-enforced-by-the-templates)):

- **Sharpened hypothesis** — the claim stated precisely enough to test.
- **Null / equivalence baseline** — what "no effect" means; for a *null* hypothesis, the
  equivalence bounds ±ε within which depth's benefit is deemed negligible (tested *for*
  equivalence via TOST, not merely "failed to reject a difference").
- **Rival hypotheses & discriminating experiments** — enumerate the competing explanations
  (for depth-null: expressivity vs. optimization vs. data-structure vs. metric-ceiling)
  and design tests that tell them *apart*, not just confirm one (strong inference).
- **Predictions** — what we expect to observe if true vs. if false, pre-specified.
- **Decision rule (pre-specified analysis)** — metric, threshold, **number of seeds**, and
  the estimator/CI procedure (`rliable`-style IQM + stratified bootstrap CIs), specified
  precisely enough to constitute a *severe* test (one the claim would probably fail if
  false).
- **Power / minimum detectable effect** — how many seeds/datasets are needed to detect an
  effect of size ε, so a null result rests on a test that *could* have found an effect.
- **Experiments needed (in principle)** — datasets, arms, metrics — *what* must be run,
  not yet *how* (that is `design.md`).

### `design.md` + `plan.md` — the engineering (delegated)

Produced by `superpowers:brainstorming` (→ `writing-plans`) *in service of* `strategy.md`:
how to realize the required experiments in the orchestration system — which benchmark
experiment specs (TOML under `benchmarks/experiments/`), configs, and groups.

### `findings.md` — the results (after runs)

- **Confirmatory result** — evaluated strictly against `strategy.md`'s pre-specified
  decision rule (plus the equivalence test for null claims). Reported first.
- **Exploratory observations** — anything *not* pre-specified, explicitly labeled as
  exploratory so it is never mistaken for a confirmatory finding (anti-HARKing).
- Result tables inside `render` **managed blocks** (auto-updating from committed result
  JSON) with links to run-hashes.
- **Threats to validity (adversarial pass)** — a section that argues *against* the
  verdict: what would make it wrong, which confound remains. Optionally produced by a
  red-team subagent before the verdict is locked.
- **Environment & provenance** — git SHA, `uv.lock` hash, package/CUDA/hardware, seeds
  (drawn from the orchestration run record), so every number is reproducible.
- **Verdict** — supported / refuted / inconclusive, backed by the above.
- **Deviations** from strategy/plan.

### Document ↔ stage lifecycle

| Document | Stage | Owner |
|---|---|---|
| `hypothesis.md` (claim + frontmatter) | capture | skill (interview) |
| `strategy.md` (science) | strategize | skill (scientific-reasoning dialogue) |
| `design.md`, `plan.md` (engineering) | design | delegated to `superpowers:brainstorming` |
| `findings.md`; `verdict`/`status` in `hypothesis.md` | record | skill |

The hypothesis is the durable **question**; the strategy is **how we decide the answer**;
the design is **how we run it**; the findings are **what we found** — recorded in that
order, never presuming a later stage.

## Scientific rigor (enforced by the templates)

The document structure is only as good as the thinking it captures, so the `strategy.md`
and `findings.md` templates **bake in established best practice** rather than relying on
in-the-moment discipline — every hypothesis inherits it. Citable references are collected
in [methodology-references.md](../../research/methodology-references.md) (the methodology
basis for the paper).

- **Anti-HARKing** — confirmatory results (pre-specified in `strategy.md`) are reported
  separately from labeled exploratory ones in `findings.md`.
- **Strong inference** — `strategy.md` enumerates rival explanations and designs
  experiments that discriminate between them, not merely confirm one.
- **Severe, pre-specified tests** — the decision rule fixes metric, threshold, seed count,
  and the estimator/CI procedure (`rliable`-style IQM + stratified bootstrap) *before*
  running.
- **Null-claim rigor** — for null hypotheses (the depth-null flagship), power / minimum-
  detectable-effect planning and equivalence bounds (TOST), so "no effect found" rests on
  a test that could have found one.
- **Adversarial review** — a threats-to-validity pass argues against the verdict before it
  is locked.
- **Complete provenance** — environment + lockfile + run-hashes make every number
  reproducible.
- **No file drawer** — refuted/inconclusive hypotheses are retained in the index.

These are standards the templates enforce; the depth-null dogfood (below) is where they
get exercised first — it is a null claim, so equivalence + power apply directly.

## Skill process (staged, resumable)

Capturing a claim is a complete act on its own — you can log an `open` hypothesis and walk
away; later stages happen in later sessions. The skill is a small state machine over the
hypothesis folder: it detects the current stage from `status` and which documents exist,
or takes an explicit stage.

**capture** *(new)*
1. Interview for the claim only — statement, intuition, rationale, tags. *Deliberately
   does not ask how it will be tested.*
2. Create `docs/research/hypotheses/<date>-<slug>/hypothesis.md` from the template; set
   `status: open`; append the index row.

**strategize** *(existing `open` hypothesis, no `strategy.md` yet)*
3. Run the guided scientific-reasoning dialogue — null, predictions, confounds, required
   evidence, decision rule, experiments-in-principle — and write `strategy.md`. This is
   the skill's own work, *not* delegated: the science precedes the engineering.

**design** *(has `strategy.md`)*
4. Hand off to `superpowers:brainstorming` with two instructions: (a) the subject is "how
   to build and run the experiments that `strategy.md` requires"; (b) store outputs in this
   folder — `design.md` and (via `writing-plans`) `plan.md`. Brainstorming's
   spec-location override makes this a clean handoff, not a fork of that skill.

**record** *(experiments have run)*
5. Execute via the orchestration system (`mononet-bench run|reconcile` on the experiments
   the design defined) — outside this skill, referenced by it.
6. Write `findings.md` (tables via `render` managed blocks), evaluate against the decision
   rule, set `verdict` + `run_hashes` and `status` in `hypothesis.md`, update the index.

This makes `testing-hypothesis` a **process/orchestrator skill** that owns the scientific
stages and composes `superpowers:brainstorming` (→ `writing-plans`) for the engineering.
The resumable-stage re-entry and the science-before-engineering ordering are its crux.

## Reproducibility linkage

The record is "sufficient for reproducibility" by chaining into the orchestration spec's
provenance — no number stranded:

- `strategy.md` experiments → the benchmark experiment specs (TOML under
  `benchmarks/experiments/`) realized in `design.md`.
- `findings.md` tables → `render` managed blocks bound to those experiments' committed
  result JSON.
- `verdict.run_hashes` → the frozen `benchmarks/results/.runs/<run-hash>.json` records
  (git SHA, per-spec provenance-hashes, device map, durations).

A reader walks hypothesis → strategy → experiment spec → provenance-hash → exact code +
data version → re-run. That end-to-end chain is why this skill is built on the
orchestration system.

## Status lifecycle

`open → {supported | refuted | inconclusive}`, plus `superseded` when a later hypothesis
replaces it. The verdict and its `run_hashes` are recorded together, so a status change is
always backed by provenance. Refuted and inconclusive hypotheses are **kept, never
deleted** — negative results are part of the record (guarding against the file-drawer
effect and against re-running dead ends) and stay in the index alongside the rest.

## Dogfood / migration

The depth-null thesis currently in agent memory
(`depth-null-in-constrained-monotone-nets`) is the natural first record: migrate it to
`docs/research/hypotheses/2026-07-1x-depth-null/`. Its empirical claim and rationale seed
`hypothesis.md`; its candidate explanations (expressivity vs. optimization vs. data
structure vs. metric ceiling) and the AUC-recheck note are exactly the confounds and
required-evidence content of `strategy.md`. Reduce the memory entry to a pointer at the
folder. This validates the format on a real, in-flight hypothesis.

## Testing / validation

Skills are process documents, so validation is dogfooding + structural checks:

- **Dogfood** the four stages on the depth-null migration: capture writes a valid
  `hypothesis.md`; strategize writes `strategy.md`; design hands off and lands
  `design.md` + `plan.md` in the folder; record writes `findings.md` and updates verdict +
  index.
- **Frontmatter validity**: `hypothesis.md` frontmatter parses as YAML with the required
  creation-stage keys; a small check (reused by the optional index regeneration) flags a
  malformed record.
- **Index consistency**: every hypothesis folder has an index row and vice versa.

## Out of scope / YAGNI

- A paper-outline document (the reverse hypothesis→paper mapping) — a later artifact,
  outside this skill.
- Automated evaluation of the decision rule against results — verdicts are recorded by a
  human; auto-evaluation is a possible later enhancement.
- Any change to the `mononet` wheel.

## Follow-ups (to become GitHub issues)

- Optional `render`-integrated index regeneration from `hypothesis.md` frontmatter, so
  `docs/research/hypotheses/README.md` cannot drift from the records.
- Auto-evaluate `strategy.md`'s decision rule against committed results to *propose* a
  verdict for human confirmation, once several hypotheses exist.
- Paper-outline document format that cites hypotheses by `label`, when write-up begins.
