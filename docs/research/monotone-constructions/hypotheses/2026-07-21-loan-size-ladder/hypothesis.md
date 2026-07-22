---
status:
  level: hypothesis
  id: 2026-07-21-loan-size-ladder
  verdict: refuted            # signed off by Davor Runje 2026-07-21
  readiness: resolved
  signed-off-by: Davor Runje
  signed-off-date: 2026-07-21
  evidence: ['run-ref://benchmarks/results/size-ladder/loan.json']
  covers: []
  load-bearing: false
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Hypothesis: Monotone depth's benefit grows with training-set size (loan size ladder).

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

Monotone depth's benefit grows with training-set size (loan size ladder).

## Why it matters

A size-dependent crossover would reconcile the depth-neutral tabular result with a possible large-data regime.

## What confirmation vs. refutation looks like

- **Confirming:** Deep band's IQM beats shallow as n_train grows along the ladder.
- **Refuting:** The deep band stays neutral-to-worse across all rungs of the ladder.

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md` (engineering backend).
