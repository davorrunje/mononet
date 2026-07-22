---
status:
  level: hypothesis
  id: 2026-07-21-hp-search-sensitivity
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

# Hypothesis: Per-flavor verdicts are hyperparameter-budget-dependent (sensitivity curves quantify how the winner shifts with trial budget).

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

Per-flavor verdicts are hyperparameter-budget-dependent (sensitivity curves quantify how the winner shifts with trial budget).

## Why it matters

If verdicts flip with budget, single-budget flavor comparisons are unreliable; the sensitivity curve is the honest reporting unit.

## What confirmation vs. refutation looks like

- **Confirming:** Sensitivity curves show the flavor ranking changing with trial budget (G-metric > 0 past trial 1).
- **Refuting:** Rankings are budget-stable (a single budget suffices).

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-15-hp-search-sensitivity-curves-design.md` (engineering backend).
