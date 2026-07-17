# scholar

**A scientific research-workflow plugin for Claude Code.** `scholar` is the
*scientific* counterpart to [`superpowers`](https://github.com/) (which handles
*engineering*): a uniform, git-native way to go from idea → literature →
hypothesis → test → publish-decision → paper → thesis, so a researcher and their
collaborators work the same way and re-derive neither the workflow nor its rigor
per project.

> **Status:** design stage. The full design lives as specs, decision records, and
> verified-source digests (see [Design & reasoning](#design--reasoning));
> implementation is pending. This README is staged in the `mononet` repo and
> moves to the standalone public `scholar` plugin repo when it is created.

## The two guiding principles

Everything in `scholar` sits under two principles:

1. **Assistants, not researchers (agency).** The skills keep the accounts, advise
   as a mentor, and discuss as a colleague — but they do **not** perform
   independent research or make material scientific decisions. Every material
   decision (is a hypothesis confirmed/refuted, is a result real, is a paper worth
   publishing, what the thesis claims, is it defensible) is the researcher's,
   recorded with a **named human sign-off**. You author; the skill drafts. You
   cannot "run" the workflow to produce a paper or thesis — you drive it.
2. **You must understand it (understanding).** Every material claim, decision, and
   method must be understood to the standard a good mentor or reviewer expects.
   `scholar` verifies *and builds* that understanding through Socratic grilling and
   teaching (the `grill` skill), and will not let work advance past a gap silently
   — including grilling the *why* behind the methodology, to prevent cargo-cult
   rigor.

Both are grounded in the literature (authorship/accountability norms, human
oversight, automation-bias and metacognition research) — see the digests.

## How it is organized

A single object×action shape at **three nested levels**:

| Level | generate | resolve |
|---|---|---|
| **hypothesis** (within a paper) | `hypothesis-exploration` | `hypothesis-testing` |
| **paper** (portfolio) | `paper-exploration` | `paper-synthesis` |
| **thesis** (optional top) | `thesis` (framing) | `thesis` (synthesis) |

Each resolve skill drives one candidate through **science-before-engineering**
staged documents (e.g. hypothesis → strategy → design/plan → findings), delegating
the *engineering* (design, plans, code) to `superpowers`.

**Three shared capabilities** the pipeline draws on:

- **`literature`** — `scout` (mine the citation graph for leads) and `position`
  (related-works synthesis / precedent), over a bibliography (CSL-JSON) + a triage
  sidecar recording your decisions about each paper.
- **`dataset`** — a registry + tiered retrieval (committed / auto-fetch /
  gated) + a per-project private mirror (rclone) with SHA-256 fixity and Gebru
  datasheets.
- **experiment backend** — a pluggable *contract* (run / evidence / tables /
  is-current); each repo supplies its own implementation, so `scholar` never
  depends on a specific runner.

**Two cross-cutting skills:**

- **`progress`** — status lives in each artifact's frontmatter; a generated
  dashboard rolls it up *semantically* (coverage + blockers, never a score;
  refuted = done).
- **`grill`** — the Socratic tutor-examiner (above), with author-selectable
  mentor/reviewer personas (never inferred from personality).

**Onboarding:** `research-init` scaffolds a fresh repo (`init`) or backfills an
existing one (`adopt`).

## Composition

`scholar` is scientific-workflow only. It **delegates engineering to
`superpowers`** and depends only on capability *contracts* (experiment backend,
`literature`, `dataset`) — so it is domain-neutral and reusable across any
research repo. It favors a light-dependency, git-native posture (plain-text
registries, committed provenance; no external tracker as source of truth).

## Design & reasoning

The design is captured in three complementary layers (all migrate here with the
plugin):

- **Specs** — the *what*: a parent meta-spec + four sub-specs (lifecycle;
  literature; dataset; substrate + experiment contract).
- **Decision log** ([`decisions/`](decisions/)) — the *why*: MADR-style ADRs, each
  with the options considered and the **rejected alternatives and why**.
- **Reference digests** ([`references/`](references/)) — the *evidence*: verified
  primary-source digests behind each skill and principle.

This record is intended to seed a blog post / paper explaining the skills and
their rationale — ideally written *using* `scholar` itself.

## License

Apache-2.0 (matching `mononet`).
