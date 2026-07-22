---
status:
  level: hypothesis
  id: 2026-07-21-alternate-construction-base
  verdict: refuted            # PROPOSED — awaiting the author's named sign-off
  readiness: pending        # 'pending' until signed; 'resolved' once signed
  signed-off-by: null          # REQUIRED — set to the author + date to make the verdict real
  signed-off-date: null
  evidence: ['run-ref://benchmarks/results/alternate-base/']
  covers: []
  load-bearing: false
  understanding: {status: pending, unresolved: []}
  blockers:
    - analysis in progress on an open draft PR; verdict trending 'refuted' but not final — do not sign yet
  last-updated: 2026-07-21
---

# Findings: 2026-07-21-alternate-construction-base

## Results

- **Evidence (run-refs):** committed result JSONs listed in the `evidence`
  frontmatter above. Numbers are not hand-copied here; regenerate tables from the
  cited run-refs via the experiment backend's `tables` capability.

## Proposed verdict (PENDING the author's sign-off)

**`refuted`** (trending — NOT yet signed) — The 4-flavor shallow bake-off shows alternate is not superior at <=4 plain layers; a single layer collapses the free branch's expressivity to convex (originating spec). Its regime is depth, not shallow.

**Status (author, 2026-07-21):** the analysis is not finished — it is on an open
draft PR and will conclude this way. Left unsigned until the analysis completes.

## Sign-off

Retroactive verdicts are still verdicts (research-init guardrail). This verdict is
**not real** until the author sets `signed-off-by` + `signed-off-date` in the
frontmatter. Proposed by adopt on 2026-07-21; awaiting confirmation.
