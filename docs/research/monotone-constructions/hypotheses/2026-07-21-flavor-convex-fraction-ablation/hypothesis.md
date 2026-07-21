---
status:
  level: hypothesis
  id: 2026-07-21-flavor-convex-fraction-ablation
  verdict: inconclusive            # PROPOSED by adopt from committed evidence — see findings.md
  readiness: pending
  signed-off-by: null          # REQUIRED — retroactive verdict is not real until the author signs
  signed-off-date: null
  evidence: ['run-ref://benchmarks/results/alternate-base/']
  covers: []
  load-bearing: false
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Hypothesis: convex_fraction is a material hyperparameter whose setting changes flavor performance.

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

convex_fraction is a material hyperparameter whose setting changes flavor performance.

## Why it matters

If convex_fraction matters, it must be tuned/reported per flavor; if not, it can be fixed and dropped from the search.

## What confirmation vs. refutation looks like

- **Confirming:** A monotone/observable effect of convex_fraction on the headline metric across flavors.
- **Refuting:** convex_fraction has no material effect (can be fixed).

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-14-flavor-ablation-benchmark-design.md` (engineering backend).
