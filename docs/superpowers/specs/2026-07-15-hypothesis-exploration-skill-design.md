# Hypothesis-exploration skill

**Status:** design
**Date:** 2026-07-15
**Author:** Davor Runje
**Scope:** a new Claude Code skill for *generating* research hypotheses, plus a backlog
artifact. Repo tooling/process — no impact on the `mononet` wheel.

## Problem

The [hypothesis-testing skill](2026-07-15-hypothesis-testing-skill-design.md) handles the
*confirmatory* loop — take one hypothesis from claim to verdict. Nothing systematically
does the *generative* half: turn the accumulated corpus of tested hypotheses (and their
surviving rival explanations, anomalies, and exploratory observations) into the next
candidates worth testing, and give loose ideas a durable home before they are formalized.
Today ideas are lost, or jump straight to ad-hoc experiments with no record of why they
were chosen.

We want a skill for the **abductive** half of the research loop: read the corpus and the
paper, propose prioritized candidate hypotheses, park ideas that aren't ready, and feed
accepted ones into the pre-registered testing skill — without ever blurring exploration
and confirmation.

## Relationship to the other skills

One of the four composable skills in the **research-workflow family** — the full matrix,
the two nested flywheels, and the exploration/testing/synthesis firewall live in
[docs/research/README.md](../../research/README.md):

- `hypothesis-exploration` — generate candidate hypotheses · `hypothesis-testing` — test one to a verdict
- `paper-exploration` — propose application papers · `paper-synthesis` — assemble a paper spine

**This skill, `hypothesis-exploration`,** is the generative half — abduction (Peirce): it reads the corpus and proposes prioritized candidate hypotheses, which only `hypothesis-testing` may confirm. It builds on the benchmark orchestration spec
([2026-07-15-benchmark-experiment-orchestration-design.md](2026-07-15-benchmark-experiment-orchestration-design.md))
for reproducible results, and is grounded in
[hypothesis-exploration-references.md](../../research/hypothesis-exploration-references.md).

## Multi-paper scoping

These skills operate on a **research project** — a paper with a uniform internal layout —
selected by a `paper-id` resolved through the registry `docs/research/papers.md`
(`paper-id → root · kind (main|application) · status`). Every folder path in this spec
(`hypotheses/`, `backlog.md`, `paper/`) is **relative to the resolved paper root**; the
examples show the main paper for concreteness.

- **Main paper** root: `docs/research/main/` (its experiments live in `benchmarks/`).
- **Application papers** root: `applications/<slug>/` — co-located with their code,
  results, and `paper/`, extending the PR #116 `applications/` convention. These sit
  outside the `docs/` tree, so they are not part of the Sphinx build at all.
- `paper-id` defaults from context — the `applications/<slug>/` you are working in, else
  the registry's designated main paper — and is otherwise prompted.
- **Application papers read the main paper read-only**: generation, dedup, and thesis-relevance for an application paper draw on the main paper's corpus as well as the application's own.

## Design principles

1. **Exploration proposes, testing disposes.** Everything this skill emits is
   *exploratory* until it passes a fresh, pre-registered test in `hypothesis-testing`
   (Wagenmakers et al. 2012). The skill never confirms its own output — that firewall is
   the whole reason it is a separate skill.
2. **Origin-agnostic pipeline.** A candidate you propose and a machine-generated one flow
   through the same path; the generation *mechanism* is a swappable front-end, not the
   core.
3. **Advisory prioritization.** The rubric ranks and suggests; the human picks what to
   test, and may promote several to test concurrently.
4. **No idea lost, no file drawer.** The backlog durably holds parked and proposed ideas;
   rejected candidates are kept with a reason, never silently deleted.

## Section 1 — Layout & artifacts

```
.claude/skills/hypothesis-exploration/
├── SKILL.md          # frontmatter + "When to use" + the verbs (park/generate/rank/promote/drop)
└── BACKLOG.md        # backlog entry format (as create-issue has STYLE.md)

docs/research/main/backlog.md   # the living idea queue (git-tracked)
```

A **backlog entry** is lightweight — *not* a hypothesis, just enough to not lose an idea
and to prioritize it later:

```
- id: <slug>
  title: <one line>
  claim: <one-sentence statement>
  origin: user | anomaly | boundary | mechanism | contradiction | generalization | negation | eda
  added: YYYY-MM-DD
  priority: {score, rationale}      # from the rubric; advisory
  status: parked | ranked | promoted | dropped
  link: <hypothesis folder if promoted>
  note: <optional; drop reason if dropped>
```

`promoted` links to the hypothesis folder it became; `dropped` keeps the reason.

## Section 2 — The origin-agnostic pipeline

One path for every candidate, whatever its source:

**origin → novelty/dedup → prioritize (advisory) → human gate → emit as `open` hypothesis**

The skill's verbs:

- **park** — stash an idea (yours or generated) in `backlog.md`, near-zero ceremony. This
  is the common entry: you have an idea you don't want to pursue right now.
- **generate** — produce candidates from the corpus. You choose the **mechanism at
  invocation**, based on corpus size and time available:
  - *lean interactive* — walk the generation moves as a guided checklist, propose a small
    ranked slate in one pass (default for a small corpus);
  - *fan-out + tournament* — subagents generate in parallel (one per move/lens), a pairwise
    ranking tournament selects the top few (the scale-up for a large corpus; Gottweis et
    al. 2025);
  - *single-pass* — one-shot ranked slate, no explicit checklist.
  Generated candidates land in the backlog.
- **rank** — score backlog entries with the rubric (Section 3).
- **promote** — accept one *or several* entries; each becomes a hypothesis folder via
  `hypothesis-testing` **capture**; the entry is marked `promoted` with a link.
- **drop** — reject an entry, keeping the reason (file-drawer discipline).

A hypothesis you already believe in skips `generate`/backlog and goes straight to
`promote` → capture.

## Section 3 — Generation moves & prioritization rubric

**Generation moves** (each grounded in `hypothesis-exploration-references.md`), applied by
the checklist or the fan-out lenses:

- **anomaly** — explain a finding that violated expectation (Kuhn 1962); the depth-null is
  the standing example.
- **boundary / scope** — probe *where* a confirmed effect stops holding (Busse et al. 2017).
- **mechanism** — represent claimed effects as a DAG and read off untested edges /
  confounders (Pearl 2009).
- **contradiction** — reconcile two findings in tension.
- **generalization / transfer** — carry a confirmed relation to a new target by structural
  analogy (Gentner 1983).
- **negation-of-refuted** — invert a refuted hypothesis into its testable complement.
- **EDA-as-generator** — mine exploratory observations for patterns (Tukey 1977), firewalled
  as exploratory (Wagenmakers et al. 2012).

All are abductive (Peirce; Lipton 2004) — proposing the "loveliest" explanation while
flagging a "best of a bad lot."

**Prioritization rubric** (advisory tiers; expected information gain as the conceptual
anchor, not a computed integral — Lindley 1956; Chaloner & Verdinelli 1995):

- **Information gain** — how much would the candidate's result *discriminate the standing
  rival explanations* or resolve an open question? (the strong-inference criterion).
- **Testability & cost** — can it be run cheaply in the orchestration system?
- **Paper relevance** — does it support/close a gap in the current paper thesis?

The rubric ranks the slate; the human decides which (one or several) to promote.

## Section 4 — Corpus scope

The skill reads, for both generation and dedup:

- **All hypothesis records** — findings' surviving rival explanations, anomalies, and
  exploratory observations; open hypotheses.
- **The backlog** — to avoid re-proposing a parked/dropped idea.
- **Paper artifacts** (outline/draft) **when present** — so gaps in the paper narrative
  seed generation and ground the "paper relevance" rubric dimension. Forward-compatible:
  read if they exist (produced by the future `paper-synthesis` skill), skipped otherwise.

Dedup guards against re-proposing anything already open, tested, parked, or dropped.

## Section 5 — Relationships & the flywheel

`promote` hands off to `hypothesis-testing` capture; the paper outline this skill reads is
produced by the future `paper-synthesis` skill. Together the three close the research
flywheel:

**paper gaps → explore (propose) → test (confirm/refute) → record → synthesize → new gaps**

Three skills, one shared `docs/research/` corpus, one firewall between exploration and
confirmation.

## Section 6 — Skill process (resumable, mode-selected)

`SKILL.md` dispatches on the verb (or infers intent):

1. **park** — write a backlog entry from a short idea; done.
2. **generate** — pick the mechanism; ingest the corpus (Section 4); apply the moves;
   dedup; score with the rubric; write the ranked candidates to the backlog.
3. **rank** — (re)score existing backlog entries, e.g. after new findings land.
4. **promote** — for each accepted entry, invoke `hypothesis-testing` capture to create the
   hypothesis folder, then mark the entry `promoted` with the link.
5. **drop** — mark an entry `dropped` with a reason.

Each verb is a complete act; the skill is re-entered as the research proceeds.

## Grounding & dogfood

- Every move and rubric line cites `hypothesis-exploration-references.md`.
- **Dogfood**: run `generate` against the depth-null corpus. Its four rival explanations
  (expressivity / optimization / data-structure / metric-ceiling) and the AUC-recheck note
  should yield a *boundary* candidate ("where does the null break — width, dimensionality,
  target smoothness?") and a *mechanism* candidate ("*why* no depth benefit — is it an
  optimization failure or a genuine expressivity ceiling?"), each parked with a priority
  rationale.

## Testing / validation

- **Dogfood** the verbs end to end on the depth-null corpus: `generate` produces a ranked
  backlog slate; `promote` creates a hypothesis folder via capture and links it; `drop`
  records a reason; `park` adds a standalone idea.
- **Dedup**: a candidate matching an existing open/tested/parked hypothesis is flagged, not
  re-proposed.
- **Firewall**: emitted candidates carry `status: open` and never a verdict; the skill
  writes no `findings.md`.

## Out of scope / YAGNI

- **Formal numeric EIG** — start with the rubric (EIG as anchor); a computed proxy is a
  later enhancement.
- **Fan-out + tournament as the default** — available as a chosen mechanism, but the lean
  checklist is the default until the corpus is large; the tournament is the documented
  scale-up.
- **`paper-synthesis`** — its own later spec; this skill only *reads* the paper artifacts
  it will produce.
- Any change to the `mononet` wheel.

## Follow-ups (to become GitHub issues)

- Formal expected-information-gain proxy for candidate scoring, once the corpus and the
  outcome model are rich enough to make it meaningful (Foster et al. 2019).
- Fan-out + tournament generation mode implementation when a `generate` run over the full
  corpus is too large for the lean checklist (Gottweis et al. 2025).
- `paper-synthesis` skill spec — the third skill and the source of the paper outline this
  skill consumes.
