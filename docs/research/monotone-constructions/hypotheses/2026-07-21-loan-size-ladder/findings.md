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

# Findings: 2026-07-21-loan-size-ladder

## Results

- **Evidence (run-refs):** committed result JSONs listed in the `evidence`
  frontmatter above. Numbers are not hand-copied here; regenerate tables from the
  cited run-refs via the experiment backend's `tables` capability.

## Proposed verdict (PENDING the author's sign-off)

**`refuted`** — The loan size ladder shows the deep band neutral-to-worse across sizes; no size-dependent crossover in favor of depth on loan.

## Sign-off

Retroactive verdicts are still verdicts (research-init guardrail). This verdict is
**not real** until the author sets `signed-off-by` + `signed-off-date` in the
frontmatter. Signed off by Davor Runje on 2026-07-21.
