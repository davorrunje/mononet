---
status:
  level: hypothesis
  id: 2026-07-21-composition-aware-deep-init
  verdict: confirmed            # signed off by Davor Runje 2026-07-21
  readiness: resolved
  signed-off-by: Davor Runje
  signed-off-date: 2026-07-21
  evidence: ['run-ref://benchmarks/results/deep-init/trainability.json']
  covers: []
  load-bearing: true
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Findings: 2026-07-21-composition-aware-deep-init

## Results

- **Evidence (run-refs):** committed result JSONs listed in the `evidence`
  frontmatter above. Numbers are not hand-copied here; regenerate tables from the
  cited run-refs via the experiment backend's `tables` capability.

## Proposed verdict (PENDING the author's sign-off)

**`confirmed`** — deep-init trainability results + the absolute-init spec (init problem pre-registered and confirmed) show composition-aware init rescues deep stacks that collapse under legacy relu/softplus init.

**Follow-on claim to isolate (not yet tested):** the Sartor (`split`) vs. Runje
(`mixed`) accuracy gap is hypothesised to be an initialization artifact — a
matched-init ablation should show `mixed` closing the gap under composition-aware
init. That ablation is its own hypothesis for `hypothesis-testing`; this finding
only confirms the trainability rescue.

## Sign-off

Retroactive verdicts are still verdicts (research-init guardrail). This verdict is
**not real** until the author sets `signed-off-by` + `signed-off-date` in the
frontmatter. Signed off by Davor Runje on 2026-07-21.
