---
status:
  level: paper
  id: cmnn-multibackend
  verdict: null
  readiness: drafting
  signed-off-by: null
  signed-off-date: null
  evidence: []
  covers: [aim-1]
  load-bearing: null
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Pitch: A multi-backend implementation of Constrained Monotonic Neural Networks

*Retroactive pitch written by research-init (adopt) from the sub-project A/B/C
specs. PROPOSED framing.*

## Central claim

`mononet` is a faithful, cross-backend-equivalent implementation of the CMNN
construction (Runje & Shankaranarayana 2023) that reproduces the paper's
Tables 1 & 2 under a standard, documented protocol.

## Contribution

A single installable package exposing the construction across **three backends**
(PyTorch, JAX/Flax NNX, Keras 3), each asserted equivalent to a NumPy reference
within fixed tolerance by a committed cross-backend equivalence harness — plus a
reproduction of the original results with modern tooling.

## Target venue + bar

Reproducibility / ML-software track (e.g. JMLR MLOSS, a reproducibility venue).
Bar: numerical equivalence across backends and reproduction within paper error
bars under matched tuning.

## Load-bearing hypotheses

- `2026-07-21-reproduce-cmnn-tables` — reproduction of the CMNN benchmark
  accuracies — load-bearing: yes.

<!-- Engineering (design/plan/implement of the package + harness) is recorded in
     the engineering backend: docs/superpowers/specs & plans for sub-projects
     A/B/C (2026-06-27-A-core-algorithm-and-backends-design.md,
     2026-05-22-B-paper-reproduction-design.md, 2026-05-22-C-extended-benchmarks-design.md). -->
