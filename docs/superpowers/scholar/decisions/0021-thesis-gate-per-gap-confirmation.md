# ADR-0021: Thesis defensibility gate — per-gap acknowledged confirmation

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The grill guardrail is everywhere else a stop-and-confirm with a single logged
override (ADR-0015), to honor agency. The thesis defense is the highest-stakes,
least-reversible material decision; should its gate be special?

## Decision drivers

- Agency: the human must always be able to proceed; the AI must not grade a novel
  claim's substance or decide a gap is "critical" (ADR-0003, ADR-0015).
- Stakes: waving a thesis through with known gaps should be hard and on record.

## Considered options

1. Uniform stop-and-confirm (single blanket override), same as other gates.
2. Hard block until no unaddressed gaps.
3. **Heavier confirmation: per-gap acknowledged sign-off** — not a hard block, but
   the author must explicitly acknowledge *each* surfaced gap in writing.

## Decision

Option 3. The thesis gate stays non-blocking (human decides) but escalates from a
single override to a **per-gap logged acknowledgement**.

## Consequences

- Waving through a thesis is deliberate and fully on record, per gap.
- The AI never adjudicates "critical" — it surfaces gaps; the human acknowledges.
- Slightly more friction at exactly the highest-stakes gate (intended).

## Rejected alternatives

- **Hard block** — forces the AI to decide a gap is critical and to block the
  human, violating agency + the non-grading rule.
- **Single blanket override** — too frictionless for the biggest, least-reversible
  decision.

## Links

sub-spec 1 §6, §10; meta-spec §2.1–§2.2; ADR-0015.
