---
status:
  level: hypothesis
  id: 2026-07-21-alternate-construction-base
  verdict: refuted            # PROPOSED by adopt from committed evidence — see findings.md
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

# Hypothesis: Tuned `alternate` beats the `mixed`/`split` incumbents at <=4 plain (non-residual) layers.

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

Tuned `alternate` beats the `mixed`/`split` incumbents at <=4 plain (non-residual) layers.

## Why it matters

If alternate had a shallow-depth edge it would be the default recommendation; its actual regime matters for the methods paper.

## What confirmation vs. refutation looks like

- **Confirming:** alternate's IQM delta CI clears zero vs both incumbents at <=4 plain layers.
- **Refuting:** alternate is not superior at shallow plain depth (its advantage, if any, is at depth).

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-14-alternate-base-result-design.md` (engineering backend).
