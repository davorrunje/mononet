---
status:
  level: hypothesis
  id: 2026-07-21-reproduce-cmnn-tables
  verdict: confirmed            # signed off by Davor Runje 2026-07-21
  readiness: resolved
  signed-off-by: Davor Runje
  signed-off-date: 2026-07-21
  evidence: ['run-ref://benchmarks/results/phase2/', 'run-ref://benchmarks/results/alternate-base/loan.json', 'run-ref://benchmarks/results/alternate-base/heart.json', 'run-ref://benchmarks/results/alternate-base/compas.json', 'run-ref://benchmarks/results/alternate-base/blog.json', 'run-ref://benchmarks/results/alternate-base/auto.json']
  covers: []
  load-bearing: true
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Hypothesis: mononet reproduces the CMNN paper Tables 1 & 2 accuracies across the benchmark datasets.

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

mononet reproduces the CMNN paper Tables 1 & 2 accuracies across the benchmark datasets.

## Why it matters

Load-bearing for the cmnn-multibackend paper: the implementation is only credible if it reproduces the published results under a documented protocol.

## What confirmation vs. refutation looks like

- **Confirming:** Per-dataset headline metrics fall within the paper's reported range under matched tuning across loan/heart/compas/blog/auto.
- **Refuting:** Systematic deviation beyond the paper's error bars on one or more datasets.

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-06-28-benchmark-foundation-and-reproduction-design.md` (engineering backend).
