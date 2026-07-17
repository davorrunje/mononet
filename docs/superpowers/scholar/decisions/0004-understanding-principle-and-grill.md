# ADR-0004: Understanding principle + the `grill` skill

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The dual of agency (ADR-0003): if the skills won't *do* the research, the author
must actually *understand* their work and methods to a mentor/reviewer standard —
otherwise they can't make good material decisions. Two grilling objects emerged:
the author's own work, and the methodology the skills embody (anti-cargo-cult).

## Decision drivers

- Illusion of explanatory depth: self-assessed understanding is unreliable until
  probed; grilling both reveals and (via retrieval/self-explanation) builds it.
- Following rigor rituals blindly is cargo-cult science.
- Goal is growth, not producing work the author can't defend.

## Considered options

1. **Second top-level principle + a cross-cutting `grill` tutor-examiner** that
   verifies *and teaches* understanding.
2. No explicit understanding check (rely on the author).
3. A grader that scores understanding.

## Decision

Option 1. `grill` probes → teaches (source-grounded) → re-probes; self-invoked
and fired as an automatic guardrail at material-decision checkpoints.

## Consequences

- Adds a cross-cutting skill and a guardrail at the three material decisions.
- Must not assert an answer key for novel claims (see ADR-0015).

## Rejected alternatives

- **No check** — leaves the illusion of explanatory depth unaddressed.
- **Scoring** — Goodhart-prone; would grade substance the AI can't authoritatively
  judge.

## Links

meta-spec §2.2, §3.7; digest `understanding-and-grilling.md`.
