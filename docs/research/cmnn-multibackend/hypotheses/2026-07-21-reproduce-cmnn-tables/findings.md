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

# Findings: 2026-07-21-reproduce-cmnn-tables

## Results

- **Evidence (run-refs):** committed result JSONs listed in the `evidence`
  frontmatter above. Numbers are not hand-copied here; regenerate tables from the
  cited run-refs via the experiment backend's `tables` capability.

## Proposed verdict (PENDING the author's sign-off)

**`confirmed`** — phase2 committed results span loan/heart/compas/blog/auto x {mixed,split} x {plain,residual,deep}; the author must confirm each headline sits within the paper's reported band.

## Sign-off

Retroactive verdicts are still verdicts (research-init guardrail). This verdict is
**not real** until the author sets `signed-off-by` + `signed-off-date` in the
frontmatter. Signed off by Davor Runje on 2026-07-21.
