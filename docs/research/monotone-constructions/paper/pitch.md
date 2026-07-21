---
status:
  level: paper
  id: monotone-constructions
  verdict: null
  readiness: drafting
  signed-off-by: null
  signed-off-date: null
  evidence: []
  covers: [aim-1, aim-2]
  load-bearing: null
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Pitch: Monotone constructions — flavors, initialization, and depth

*Retroactive pitch written by research-init (adopt) from the active
monotone-constructions thread. PROPOSED framing.*

## Central claim

The choice of monotone construction **flavor** (`mixed` / `alternate` / `split`),
its **initialization**, and **residual depth** have measurable, characterizable
effects on trainability and accuracy — and a **composition-aware initialization**
makes deep monotone stacks trainable where the shipped init collapses.

## Contribution

A systematic bake-off and set of ablations across the three flavors, initialization
schemes, and depth — turning construction folklore into measured, ablated results
with a documented protocol and per-cell evidence.

## Target venue + bar

ML methods venue. Bar: matched-tuning comparisons across flavors with pre-registered
decision rules, seed-level variance, and isolating ablations (e.g. convex_fraction).

## Load-bearing hypotheses

- `2026-07-21-alternate-construction-base` — alternate flavor base result.
- `2026-07-21-composition-aware-deep-init` — composition-aware init rescues deep stacks — load-bearing: yes.
- `2026-07-21-deep-monotonic-residual-accuracy` — deep monotonic residual accuracy.
- `2026-07-21-monoresidual-gate-stability` — MonoResidual gate instability + fix.
- `2026-07-21-loan-size-ladder` — sample-size sensitivity (loan ladder).
- `2026-07-21-flavor-convex-fraction-ablation` — convex_fraction ablation.
- `2026-07-21-hp-search-sensitivity` — HP-search sensitivity curves.

<!-- Engineering recorded in the engineering backend: docs/superpowers/specs &
     plans (2026-07-13-monotone-constructions-init-and-ablation-design.md,
     2026-07-14-alternate-base-result-design.md, 2026-07-14-flavor-ablation-benchmark-design.md,
     2026-07-13-stage2-unified-depth-benchmark-design.md, 2026-07-03-deep-monotonic-residual-design.md,
     2026-07-05-deep-residual-accuracy-design.md, 2026-07-02-absolute-init-deep-networks-design.md,
     2026-07-13-monoresidual-gate-instability-fix-design.md, 2026-07-15-hp-search-sensitivity-curves-design.md,
     2026-07-10-loan-size-ladder-design.md). -->
