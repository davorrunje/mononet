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

# Hypothesis: The MonoResidual gate has an initialization-dependent instability that pins the depth gate, and the scaled_elu fix resolves it.

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

The MonoResidual gate has an initialization-dependent instability that pins the depth gate, and the scaled_elu fix resolves it.

## Why it matters

A pinned gate silently disables depth; characterizing and fixing it is required before any depth claim is meaningful.

## What confirmation vs. refutation looks like

- **Confirming:** The instability reproduces (g_beta pinned at ~eps, F's last-layer weights frozen) and the fix restores depth use.
- **Refuting:** The gate is stable as-shipped and depth is used without the fix.

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-13-monoresidual-gate-instability-fix-design.md` (engineering backend).
