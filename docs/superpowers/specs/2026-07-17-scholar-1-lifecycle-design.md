# scholar sub-spec 1 — Lifecycle, pipeline skills, rigor, progress & grill

**Date:** 2026-07-17
**Author:** Davor Runje
**Status:** Brainstorming output; sub-spec of the `scholar` meta-spec, pending implementation plan.

> Sub-spec of [2026-07-17-scholar-plugin-design.md](2026-07-17-scholar-plugin-design.md).
> The integrative capstone: the three-level lifecycle, the five pipeline skills,
> the rigor kit, and the two cross-cutting skills (`progress`, `grill`). Consumes
> the experiment-backend contract of
> [sub-spec 4](2026-07-17-scholar-4-substrate-and-experiment-contract-design.md) §3
> and the `literature`/`dataset` capabilities of
> [sub-spec 2](2026-07-17-scholar-2-literature-design.md) /
> [sub-spec 3](2026-07-17-scholar-3-dataset-design.md). Governed by the ⚑ agency
> (§2.1) and Understanding (§2.2) principles.
> Refines the four `2026-07-15-*` research-workflow specs (which migrate here) and
> adds the thesis level, progress, and grill. Grounding:
> [thesis-and-progress-tracking](../scholar/references/thesis-and-progress-tracking.md),
> [understanding-and-grilling](../scholar/references/understanding-and-grilling.md),
> [mentor-personas](../scholar/references/mentor-personas.md).

## 1. The three-level mirror & flywheels

One object×action shape at three nested levels (meta-spec §3.1). Each level: a
**generate** skill proposes candidates into a backlog; a **resolve** skill drives
one candidate through staged docs to a human-signed verdict/artifact. The
**firewall** holds (explore proposes, resolve disposes, synthesize reports); every
stage is human-driven.

| Level | generate | resolve | staged docs | backlog |
|---|---|---|---|---|
| hypothesis | `hypothesis-exploration` | `hypothesis-testing` | hypothesis → strategy → design/plan → findings | `backlog.md` |
| paper | `paper-exploration` | `paper-synthesis` | pitch → positioning → outline/plan → decision → sections | `portfolio-backlog.md` |
| thesis *(optional top)* | `thesis` (*framing*) | `thesis` (*synthesis*) | prospectus → aims/narrative → chapter↔paper map → kappa + defensibility | the portfolio |

Three nested loops: hypotheses accumulate into a paper; papers into a portfolio;
papers cover the thesis aims.

## 2. Pipeline skills

### 2.1 `hypothesis-exploration` (generate)

Origin-agnostic idea pipeline into `backlog.md`; verbs `park | generate | rank |
promote | drop`. Generation moves (six + EDA) and an EIG-anchored ranking rubric
(from the migrated `2026-07-15-hypothesis-exploration` spec). **Calls
`literature scout`** for citation-seeded leads (each carries provenance to the
citing paper). *Proposes only* — the human promotes.

### 2.2 `hypothesis-testing` (resolve) — science before engineering

Staged docs, in order:
1. **hypothesis.md** — free-form claim (from a promoted backlog item).
2. **strategy.md** — the *scientific* thinking, **before** any engineering: how to
   confirm/refute, rival hypotheses + discriminating tests, the rigor-kit choices
   (§4). Calls `literature position --level hypothesis` (adversarial precedent
   rapid-review: "would a reviewer say this is already known?").
3. **design.md / plan.md** — the *engineering*, **delegated to `superpowers`**
   (brainstorming → writing-plans), stored under the hypothesis folder.
4. **findings.md** — results (citing backend **run-refs**, never hand-copied
   numbers) + the **verdict** (confirmed / refuted / inconclusive), which is a
   **material decision recorded with a named human sign-off** (§2.1).

Uses the **experiment-backend contract** (sub-spec 4 §3): `run` → run-ref,
`evidence` → results + provenance stamp, `tables` → result blocks, `is-current` →
staleness. Datasets declared here are materialized/verified via `dataset`
(sub-spec 3).

### 2.3 `paper-exploration` (generate, portfolio)

Proposes application/follow-up papers into `portfolio-backlog.md` via
mechanism-transfer / limitation-driven / result-driven lenses and feasibility×interest
prioritization (from the migrated `2026-07-15-paper-exploration` spec). Defines the
`docs/research/papers.md` registry (paper-id → root + `backend:` binding). **Calls
`literature scout --level paper`.**

### 2.4 `paper-synthesis` (resolve, portfolio)

Staged docs: **pitch → positioning → outline/plan → decision → sections**.
- **positioning.md** — related-works synthesis via `literature position --level
  paper` (taxonomy + concept matrix + PRISMA log + per-branch delta + baselines).
- **outline/plan** — engineering of the paper structure, delegated to `superpowers`.
- **decision.md** — the **publish / no-go verdict**, a material decision with a
  named human sign-off, gated on accumulated hypothesis evidence + positioning
  (the paper-level mirror of a hypothesis `findings` verdict).
- **sections** — assembled from a **Toulmin-sextet claim→evidence ledger** citing
  backend run-refs; the `tables` capability writes result blocks. Drafting is
  assistive; the author authors (§2.1).

### 2.5 `thesis` (framing + synthesis, optional top level)

`framing` (occasional): define the aims/narrative and which portfolio papers
compose the thesis. `synthesis`: assemble the **kappa** framing chapter (aims,
independent related work, per-paper contribution statement, unifying discussion,
future work — *no new findings*) + appended papers, and clear the **defensibility**
gate. Roll-up target is **narrative coverage of the aims**, not paper count;
milestones live in `thesis/milestones.yml` (configurable). Monograph = degenerate
case (one paper spanning the thesis).

## 3. Rigor kit

Applied primarily in `strategy.md` (before engineering) and enforced through
`findings.md` / `decision.md`; every element is also a `grill methodology` target
(§6). Elements and the understanding each requires (from the
understanding-and-grilling digest):

- **Confirmatory vs. exploratory / preregistration** — fix analysis before
  outcomes; curbs HARKing + shrinks the forking-paths garden; does not forbid
  exploration, only mislabeling it.
- **Rival hypotheses + discriminating tests** — a competing explanation + a result
  that separates it.
- **Severity** — passing is evidence only if the test would probably have failed
  were the claim false.
- **Power / MDE** — set before data; an underpowered null is uninformative.
- **TOST / equivalence** — a non-significant p is *not* "no effect"; equivalence
  needs a pre-set bound.
- **Disclosure checklist + file-drawer** — selective reporting is what corrupts
  the record.
- **Per-dataset datasheets** — via `dataset` (sub-spec 3); Gebru-style.
- **Red-team pass** — adversarial self-review before a verdict/decision.

## 4. Staged-doc templates

The plugin ships templates for every staged doc, mirrored across levels
(hypothesis↔pitch, strategy↔positioning, design/plan↔outline/plan,
findings↔decision, +kappa). Each carries a **status frontmatter block** (§5) and
the rigor prompts (§3). Templates are proposals the author fills; the skill drafts,
the author authors.

## 5. `progress` (cross-cutting, read-only)

Per meta-spec §3.6: status lives in each artifact's **frontmatter**
(verdict/readiness/coverage + `last-updated`); `progress status <level> [id]`
reads and rolls up; `progress dashboard` regenerates `dashboard.md` as a **pure
projection** (never hand-edited).

- **Definition of done:** hypothesis = *resolved* (has a signed verdict); paper =
  *done* (its hypotheses resolved + submission-ready); thesis = *defensible* (aims
  covered + kappa through-line stated).
- **Semantic roll-up** (coverage + blockers, never a %/average): a refuted
  load-bearing hypothesis blocks its paper; verdict and readiness are distinct
  axes — **refuted = done/green, not failed**.
- **Anti-Goodhart:** surface state and gaps, never a productivity score.

## 6. `grill` (cross-cutting)

Per meta-spec §3.7: a Socratic tutor-examiner (targets `claim | cited-work |
methodology`), probe → teach (source-grounded) → re-probe; self-invoked and fired
as an automatic **guardrail** at material-decision checkpoints. Integration points:

- with `hypothesis-testing` → grill **strategy** (assumptions, entailments,
  falsifiers, rival explanations) and the **methodology** it invokes;
- with `paper-synthesis` → grill **positioning** (novelty vs. prior work; do the
  cited works support the claims — the `cited-work` target on the `literature`
  registry);
- at the **thesis defensibility gate** → a full **mock viva**.

**Guardrail semantics:** stop, surface the gap, offer to grill/teach, **record**;
the human may override, override logged — not a hard block. The **thesis
defensibility gate escalates** (ADR-0021): instead of a single blanket override,
the author must **acknowledge each surfaced gap in writing** (per-gap logged
sign-off) — still non-blocking (the AI never adjudicates "critical"), but
deliberate and fully on record at the highest-stakes decision. Never grades a novel claim's substance;
teaches the established (methodology, cited work) source-grounded. **Mentor
personas** (sounding board / critical examiner / directive editor / opt-in devil's
advocate) are author-selectable, task/stage-suggested, feedback-calibrated —
**never inferred from personality** (§3.7, mentor-personas digest).

## 7. Content layout (consumer `docs/research/`)

Per meta-spec §5: `papers.md` registry; per-paper roots with
`hypotheses/<YYYY-MM-DD-slug>/{hypothesis,strategy,design,plan,findings}.md`,
`backlog.md`, `paper/{pitch,positioning,outline,ledger,decision,sections/}`;
`portfolio-backlog.md`; optional `thesis/{kappa,aims.md,milestones.yml}`;
`literature/{references.bib,triage.yml}`; generated `dashboard.md`. Status
frontmatter on every hypothesis/paper/thesis artifact feeds `progress`; grill
transcripts + logged overrides form the accountability trail.

## 8. Agency & Understanding interlock (the material-decision checkpoints)

Three decisions are **material** and each requires a **named human sign-off**
(§2.1), with the guardrail `grill` firing beforehand (§2.2):

| Decision | Artifact | Level |
|---|---|---|
| hypothesis confirmed / refuted / inconclusive | `findings.md` verdict | hypothesis |
| publish / no-go | `decision.md` | paper |
| defensible? | thesis defensibility | thesis |

The backend stamps evidence but never adjudicates it; `is-current` reports
staleness but the human decides whether to re-run; `progress` reports coverage but
never a score; `grill` surfaces gaps but never grades substance. The human drives
every one.

## 9. Plugin vs. consumer

- **Plugin:** the five pipeline skills + `progress` + `grill`; staged-doc + kappa
  templates; the rigor kit; the firewall/flywheel logic; the persona set. Depends
  only on the capability contracts (experiment backend, `literature`, `dataset`)
  and delegates engineering to `superpowers`.
- **Consumer:** all `docs/research/` content; the `papers.md` bindings; the
  experiment-backend implementation; `.scholar/` config.

## 10. Open items

- **Thesis defensibility gate** — *resolved (ADR-0021):* non-blocking, but with
  per-gap acknowledged sign-off (heavier than the single-override used at other
  gates).
- **`design.md`/`plan.md` delegation seam** — the exact handoff to `superpowers`
  brainstorming/writing-plans (store paths under the hypothesis folder).
- **Ledger format** — the Toulmin-sextet schema for `paper/ledger`; settle in the
  plan.
- **Status frontmatter schema** — the exact fields feeding `progress` roll-up.
