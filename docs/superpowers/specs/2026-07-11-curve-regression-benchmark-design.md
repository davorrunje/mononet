# Curve-regression monotonic benchmark (Phase 3 — stub)

Status: Planned (stub — not yet brainstormed to approval)

Phase 3 of the large-dataset monotonic-depth benchmark program
([program note](2026-07-11-large-dataset-benchmark-program.md)). Placeholder
capturing scope; needs its own brainstorming → design → plan cycle.

## Why separate

The target is a monotone **curve** (per-context, multi-point), not a single
scalar row label, so it needs a different target representation and evaluation
than the Phase 1 tabular harness.

## Sketch

- **Dataset.** R&F-Inventory (Kuaishou, SIGIR '26): a large-scale, purpose-built
  monotonic dataset for reach-and-frequency inventory estimation — budget↑ →
  UV/PV↑ with diminishing returns. CC-BY, code + data at
  `github.com/pengyunshan/RF-Inventory`. The paper defines two tasks:
  single-point performance prediction and budget-performance-curve
  reconstruction.
- **New machinery.** A curve target (multiple budget points per
  targeting/scheduling/frequency context); **curve-reconstruction** error metrics
  (PV/UV MAE·RMSE) plus a **monotonicity-violation rate**; a consistency check
  against the theoretical maximum-exposure ceiling.
- **Screen + gate.** Deep/shallow max-size screen adapted to the regression/curve
  metric; gate margin expressed on the reconstruction error.

## Open questions (for its brainstorm)

- Single-point vs full-curve task first.
- Exact dataset scale and split protocol (confirm from the release).
- How the monotone layers express diminishing-returns (concave-monotone) curves.
