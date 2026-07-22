---
status:
  level: hypothesis
  id: 2026-07-21-monoresidual-gate-stability
  verdict: confirmed            # signed off by Davor Runje 2026-07-21
  readiness: resolved
  signed-off-by: Davor Runje
  signed-off-date: 2026-07-21
  evidence: ['run-ref://benchmarks/results/monoresidual-gate/scale.json', 'run-ref://benchmarks/results/monoresidual-gate/ablation.json', 'run-ref://benchmarks/results/monoresidual-gate/trap.json']
  covers: []
  load-bearing: true
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Findings: 2026-07-21-monoresidual-gate-stability

## Results

- **Evidence (run-refs):** committed result JSONs listed in the `evidence`
  frontmatter above. Numbers are not hand-copied here; regenerate tables from the
  cited run-refs via the experiment backend's `tables` capability.

## Proposed verdict (PENDING the author's sign-off)

**`confirmed`** — scale/ablation/trap results confirm the gradient-collapse trap (g_beta=0.000 at F-init under scaled_elu) and the fix; the instability and its resolution are characterized.

## Sign-off

Retroactive verdicts are still verdicts (research-init guardrail). This verdict is
**not real** until the author sets `signed-off-by` + `signed-off-date` in the
frontmatter. Signed off by Davor Runje on 2026-07-21.
