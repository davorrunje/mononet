---
status:
  level: hypothesis
  id: 2026-07-21-deep-monotonic-residual-accuracy
  verdict: inconclusive            # PROPOSED — awaiting the author's named sign-off
  readiness: pending        # 'pending' until signed; 'resolved' once signed
  signed-off-by: null          # REQUIRED — set to the author + date to make the verdict real
  signed-off-date: null
  evidence: []
  covers: []
  load-bearing: false
  understanding: {status: pending, unresolved: []}
  blockers:
    - open draft PR (deep-residual-accuracy, Stage-2) — accuracy-stage results not committed yet
  last-updated: 2026-07-21
---

# Findings: 2026-07-21-deep-monotonic-residual-accuracy

## Results

- **Evidence (run-refs):** committed result JSONs listed in the `evidence`
  frontmatter above. Numbers are not hand-copied here; regenerate tables from the
  cited run-refs via the experiment backend's `tables` capability.

## Proposed verdict (PENDING the author's sign-off)

**`inconclusive`** — No accuracy-stage results are committed (benchmarks/results/deep-residual-accuracy/ is empty; only Stage-1 trainability exists). The large-dataset screen (#115) indicates depth is neutral across all 5 datasets. Evidence lives on open branches -> see the fold-PRs follow-up.

**Status (author, 2026-07-21):** work in progress on an open draft PR; verdict
deferred until the Stage-2 accuracy results land and are committed.

## Sign-off

Retroactive verdicts are still verdicts (research-init guardrail). This verdict is
**not real** until the author sets `signed-off-by` + `signed-off-date` in the
frontmatter. Proposed by adopt on 2026-07-21; awaiting confirmation.
