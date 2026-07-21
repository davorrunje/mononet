---
status:
  level: hypothesis
  id: 2026-07-21-deep-monotonic-residual-accuracy
  verdict: inconclusive            # PROPOSED by adopt from committed evidence — see findings.md
  readiness: pending
  signed-off-by: null          # REQUIRED — retroactive verdict is not real until the author signs
  signed-off-date: null
  evidence: []
  covers: []
  load-bearing: false
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Hypothesis: Monotonic residual depth improves test accuracy on the tabular benchmarks.

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

Monotonic residual depth improves test accuracy on the tabular benchmarks.

## Why it matters

Trainability (Stage 1) is necessary but not sufficient; the paper claim needs depth to translate into accuracy.

## What confirmation vs. refutation looks like

- **Confirming:** Deep monotonic residual test metrics beat shallow under matched tuning.
- **Refuting:** Depth is neutral-to-worse for accuracy on small/medium tabular data (an honest, reportable null).

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-05-deep-residual-accuracy-design.md` (engineering backend).
