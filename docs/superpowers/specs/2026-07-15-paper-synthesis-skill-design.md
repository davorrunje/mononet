# Paper-synthesis skill

**Status:** design
**Date:** 2026-07-15
**Author:** Davor Runje
**Scope:** a new Claude Code skill that assembles the paper from the hypothesis corpus,
plus the paper artifacts it maintains. Repo tooling/process — no impact on the `mononet`
wheel.

## Problem

The [testing-hypothesis](2026-07-15-testing-hypothesis-skill-design.md) and
[hypothesis-exploration](2026-07-15-hypothesis-exploration-skill-design.md) skills produce
a growing corpus of hypotheses with verdicts and evidence. Turning that corpus into a
paper — deciding the thesis, which verdicts are headline claims vs. null results vs.
limitations, which claims still lack support — is today ad-hoc, and a paper's quoted
numbers drift from the results that back them. We want a skill that keeps a **living paper
spine**: every claim traced to the hypotheses and committed results that support it, a
gap list of what is under-supported, and a guarantee that quantitative claims stay in sync
with the evidence.

## Relationship to the other skills

Third and final skill of the research-workflow family, sharing the `docs/research/` corpus:

- `testing-hypothesis` — confirmatory (claim → verdict).
- `hypothesis-exploration` — generative (corpus → candidates).
- **`paper-synthesis`** (this spec) — synthesis (verdicts → paper spine → gap list).

It **reads** the whole hypothesis corpus and the orchestration results; it **produces**
the paper outline that `hypothesis-exploration` reads, closing the research flywheel. It is
grounded in [paper-synthesis-references.md](../../research/paper-synthesis-references.md).

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
- **Application papers read the main paper read-only**: an application paper's claims may cite main-paper results/hypotheses by `label`; `render` pulls the project's *own* results (`applications/<slug>/results/`, or `benchmarks/` for the main paper).

## Design principles

1. **Ledger-first, prose optional.** The durable core is a claim→evidence ledger; drafting
   section prose is a secondary, on-demand stage — never autonomous whole-paper generation.
2. **Every number is derived, never transcribed.** Result tables enter via `render`
   managed blocks; a claim is "supported" only when its number is recomputable from
   committed code + data (Peng 2011; Gentleman & Temple Lang 2007). This makes a staleness
   check well-defined (Manubot; Himmelstein et al. 2019).
3. **Reverse mapping.** The paper references hypotheses (by `label`), not the reverse —
   consistent with the testing skill's rule that a hypothesis never presumes a paper
   structure.
4. **Nulls are contributions.** A well-supported null (the depth-null thesis) is a
   first-class narrative role, not an omission (Rosenthal 1979).

## Section 1 — Layout & artifacts

```
.claude/skills/paper-synthesis/
├── SKILL.md          # frontmatter + "When to use" + verbs (outline/map/gaps/draft/check)
└── LEDGER.md         # claim→evidence entry format (as create-issue has STYLE.md)

docs/research/main/paper/
├── outline.md        # thesis → sections → claims (the narrative skeleton)
├── ledger.md         # claim→evidence bindings (the reproducibility spine)
└── sections/<slug>.md  # optional on-demand prose drafts
```

All Markdown, excluded from the Sphinx site build (like the rest of `docs/research/`). The
directory follows the research-compendium layout (Marwick et al. 2018): committed result
tables are the shared output layer that both the ledger and any prose read from.

## Section 2 — The claim→evidence ledger (core object)

Each claim is a **Toulmin argument sextet** (Toulmin 1958) — the grounded schema — recorded
as prior-art "micropublication" in miniature (Clark et al. 2014; Groth et al. 2010):

```
- id: <slug>                         # stable identifier (nanopublication-style addressability)
  claim: <the assertion the paper makes>          # C — claim
  section: <outline section>
  grounds: [<render table id / run-hash / findings link>, ...]   # D — the evidence
  warrant: <the analysis that turns grounds into the claim>      # W — e.g. "IQM Δ + TOST equivalence"
  backing: [<methodology-reference key>, ...]      # B — why the warrant holds
  hypotheses: [<label>, ...]                       # the hypotheses that back it
  qualifier: <force>                               # Q — encoded by status + role
  rebuttal: <conditions of exception / limitation> # R
  status: supported | thin | gap | contradicted    # computed from backing hypotheses' verdicts
  role: headline | null-result | limitation | background
```

`status` is computed from the backing hypotheses' verdicts; `thin`/`gap` are precisely a
missing or weak warrant/backing. `role` maps a verdict into a narrative slot (Schimel
2012): **supported → headline (the story's resolution); refuted/null → an explicit "we
establish X does *not* hold" contribution; inconclusive → limitation/future work.**

## Section 3 — Verbs / process

- **outline** — propose/refresh the structure: bottom-up (cluster supported hypotheses)
  *and* top-down (take your stated thesis); advisory, you shape it. Writes `outline.md`.
- **map** — (re)bind claims ↔ hypotheses ↔ evidence into the ledger; recompute each claim's
  `status` from current verdicts.
- **gaps** — emit the gap list: `thin`/`gap`/`contradicted` claims and sections with no
  backing hypotheses. Disciplined by Sandberg & Alvesson (2011) — distinguish a genuine
  *evidential* gap (a claim lacking backing) from rhetorical gap-spotting. **This list is
  what `hypothesis-exploration` reads** — the flywheel's return arc.
- **draft `<section>`** — optional: render Markdown prose for one section from its ledger
  rows, applying Gopen & Swan (1990) (claim in subject position, evidence in the stress
  position) and Schimel's (2012) narrative roles. Never the whole paper autonomously.
- **check** — staleness guard: flag any claim whose backing hypothesis verdict changed
  since it was written, and run `render --check` on the paper's result tables. A claim
  cannot silently outlive its evidence (Himmelstein et al. 2019).

## Section 4 — Consistency & the flywheel

Result tables enter via `render` managed blocks (the orchestration spec's mechanism), so
every quantitative claim traces to committed results and run-hashes; `check` mirrors
orchestration's `render --check`. The loop closes:

**paper `gaps` → `hypothesis-exploration` → `testing-hypothesis` → verdicts →
`paper-synthesis` `map`/`outline` → new `gaps`.**

Three skills, one shared `docs/research/` corpus, one firewall (exploration proposes,
testing disposes, synthesis reports).

## Section 5 — Skill process (resumable, verb-dispatched)

`SKILL.md` dispatches on the verb (or infers intent):

1. **outline** — cluster the corpus / take the thesis; write or update `outline.md`.
2. **map** — bind claims to hypotheses + evidence; recompute statuses in `ledger.md`.
3. **gaps** — emit the evidential-gap list for `hypothesis-exploration`.
4. **draft `<section>`** — optional prose for one section.
5. **check** — staleness + `render --check`; report claims whose evidence moved.

Each verb is a complete act; the skill is re-entered as verdicts accumulate.

## Grounding & dogfood

- Every schema field and verb cites `paper-synthesis-references.md`.
- **Dogfood**: run `outline` + `map` over the current corpus. The depth-null thesis should
  land as a `null-result` headline claim (warrant = IQM Δ + TOST equivalence, backing =
  the equivalence/severity references), and `gaps` should surface the mechanism question
  ("*why* no depth benefit?") as an evidential gap to feed back to exploration.

## Testing / validation

- **Dogfood** the verbs end to end: `outline` writes a thesis/section skeleton; `map`
  produces ledger rows with computed statuses; `gaps` emits an evidential-gap list;
  `draft` renders one section's prose; `check` flags a claim after its backing verdict is
  changed.
- **Consistency**: mutating a backing hypothesis's verdict makes `check` flag the claim;
  re-mapping clears it. Result tables are `render`-managed, never hand-typed.
- **Reverse-mapping invariant**: the ledger references hypotheses by `label`; no
  `hypothesis.md` gains a paper-structure field.

## Out of scope / YAGNI

- **LaTeX / Overleaf assembly** — the final manuscript is assembled externally/later from
  this Markdown spine; the skill does not emit LaTeX.
- **Autonomous whole-paper prose** — drafting is per-section, on demand.
- **Bibliography/citation management** — external (e.g. the manuscript's own toolchain).
- Any change to the `mononet` wheel.

## Follow-ups (to become GitHub issues)

- LaTeX/Overleaf export of the Markdown spine, when the manuscript toolchain is chosen.
- A `render`-integrated paper `check` in CI (fail the build when a paper claim's backing
  verdict has moved), mirroring the orchestration `render --check` pre-commit hook.
- Automated claim→hypothesis binding suggestions (propose `hypotheses:` for a new claim by
  matching statement text against the corpus), once the corpus is large.
