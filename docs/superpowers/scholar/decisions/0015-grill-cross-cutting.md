# ADR-0015: `grill` cross-cutting; guardrail stop-and-confirm; teaching; non-grading

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Given the Understanding principle (ADR-0004), how should grilling be structured,
invoked, and constrained — especially since the AI can't know the ground truth of
a novel claim. Contrasted with Matt Pocock's `grill-me` (same mechanic, inverted
epistemics: his lets the AI recommend answers; the human is ground truth).

## Decision drivers

- Author wants both self-invocation and an automatic guardrail against advancing
  work not understood.
- Author wants teaching (explanations + references), not just testing.
- Must respect agency: the AI can't fabricate a novel claim's answer or hard-block
  the human.

## Considered options

1. **One cross-cutting `grill` skill; targets claim|cited-work|methodology; teach-
   and-reprobe loop; self-invoked + guardrail; guardrail = stop/surface/offer/
   record with logged override (not a hard block); non-grading on substance.**
2. A mode baked into each pipeline skill (no standalone).
3. A grader that passes/fails understanding.
4. Hard-blocking gate at material decisions.

## Decision

Option 1. Teaches the *established* (methodology, cited work) source-grounded;
never asserts a *novel* claim's answer; settled-vs-contested calibration; depth by
stakes; anti-Goodhart. Personas per ADR-0016.

## Consequences

- Consistent stance across stages; one place to maintain.
- The guardrail records logged overrides (accountability) but the human still
  drives; the thesis gate escalates to per-gap acknowledgement (ADR-0021).

## Rejected alternatives

- **Per-skill mode** — duplicates the stance; harder to keep invariant.
- **Grader** — grades substance the AI can't authoritatively judge; Goodhart-prone.
- **Hard block** — violates agency / human-oversight-with-override.

## Links

meta-spec §3.7; sub-spec 1 §6; digest `understanding-and-grilling.md`.
