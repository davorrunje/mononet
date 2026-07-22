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

# Hypothesis: A composition-aware initialization makes deep plain monotone stacks trainable where the shipped/legacy per-layer init collapses.

*Retroactive hypothesis reconstructed by research-init (adopt) from committed
results and the originating engineering-backend spec.*

## Claim

A composition-aware initialization makes deep plain monotone stacks trainable where the shipped/legacy per-layer init collapses.

## Why it matters

Trainability of deep monotone stacks is the precondition for any depth benefit; the init is the load-bearing fix.

**Interpretation (author, 2026-07-21):** the reported advantage of Sartor et al.
(`split`) over Runje & Shankaranarayana (`mixed`) is hypothesised to be an artifact
of **sub-optimal initialization** rather than a fundamental property of the
construction. With composition-aware init, the `mixed` construction is expected to
close the gap — i.e. the Sartor-vs-Runje comparison is confounded by init. This
reframes the `split` win as an init effect, and is a load-bearing claim the
`monotone-constructions` paper should isolate with a matched-init ablation.

## What confirmation vs. refutation looks like

- **Confirming:** Deep (>=16-layer) plain stacks train (bounded loss, non-divergence) under composition-aware init while legacy init diverges/collapses for relu/softplus.
- **Refuting:** Composition-aware init shows no trainability advantage over legacy init at depth.

## Provenance

Reconstructed from committed benchmark results and the design record:
`docs/superpowers/specs/2026-07-02-absolute-init-deep-networks-design.md` (engineering backend).
