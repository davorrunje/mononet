# Residual Depth + Skip-Granularity Ablation — Idea

**Date:** 2026-06-30
**Status:** Idea (deferred — not yet brainstormed/spec'd)
**Sub-project:** C (extended benchmarks / ablations)

> Parked follow-up. Capture only — pick up via the normal brainstorm → spec → plan
> flow when scheduled. Out of scope for the protocol PR (#62), which is
> evaluation-protocol only.

## Idea

Ablate the **residual flavor's depth and skip granularity** together — they pair
naturally and test whether the dual-gated `MonoResidual` block actually buys *usable*
network depth.

## Current state

- The residual stack is `depth × MonoResidual`, and each `MonoResidual` wraps a
  **single** `MonoLinear` (`mononet/.../torch/layers.py`) — i.e. **skip every 1 layer**.
  Forward: `g_α·skip(x) + g_β·F(x)`, with `F` defaulting to one `MonoLinear`.
- `benchmarks/_common/search_spaces.py`: `depth = suggest_int("depth", 1, 4)` — the
  **same range for plain and residual**. Depth has not been widened for residual despite
  the near-identity warm start (`α`/`β` gates init to 0) that should make deeper nets
  trainable.
- On AutoMPG, tuned best configs sit at depth 1–2, so within `[1,4]` there's no evidence
  of a depth limit yet — itself a reason to test a wider range.

## Proposal (residual flavors only)

1. **Deeper range:** widen the `depth` axis (e.g. `[1, 8]`) and check whether residual
   reaches usable depths that plain cannot.
2. **Skip granularity:** add a `skip_every ∈ {1, 2}` axis. **No new primitive needed** —
   `MonoResidual.__init__` already accepts a custom `F` (module or `units→module`
   callable), so an every-2-layers block is `F = (two stacked MonoLinears)`.
3. Compare against the current skip-every-1 / shallow residual at each flavor's tuned best.

## Notes

- Search-axis + architecture change; **not** a protocol change.
- Run under the standard held-out protocol from
  [2026-06-30-standard-benchmark-protocol-design.md](2026-06-30-standard-benchmark-protocol-design.md)
  (5-fold CV HP selection, mean±std over all seeds, no best-k).
- Keep `embed_hidden` (free-feature embedding) in mind as a related untuned axis.
