# ADR-0005: Three-level mirror; thesis as a partial mirror

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

We started with two levels (hypothesis within a paper; paper within a portfolio).
The author then asked to incorporate the PhD thesis and to track progress toward
it, and to keep the structure uniform across levels.

## Decision drivers

- Uniformity across levels reduces cognitive load (the author's stated priority).
- Thesis-by-publication *is* a papers→thesis nesting (kappa binds them).
- But a thesis is not "explored" in bulk — there is one, framed once.

## Considered options

1. **Thesis as a third, top, optional level — a *partial* mirror** (synthesis +
   progress + defensibility; framing degenerate).
2. Full symmetric third level with a real thesis-exploration flywheel.
3. No thesis level; just a narrative doc + dashboard over the portfolio.

## Decision

Option 1. Thesis = optional top level; `thesis` skill (framing + synthesis);
roll-up target is narrative coverage of aims, not paper count; monograph is the
degenerate case. Cross-repo aggregation out of scope (ADR-0018).

## Consequences

- Honest asymmetry documented (no high-throughput thesis exploration).
- A repo that isn't a thesis simply omits the top level.

## Rejected alternatives

- **Full symmetric flywheel** — false symmetry; there is one thesis.
- **No level** — can't assemble the kappa or gate defensibility.

## Links

meta-spec §3.1; sub-spec 1 §2.5; digest `thesis-and-progress-tracking.md`.
