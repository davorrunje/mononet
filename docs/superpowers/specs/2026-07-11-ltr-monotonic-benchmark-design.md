# Learning-to-rank monotonic benchmark (Phase 2 — stub)

Status: Planned (stub — not yet brainstormed to approval)

Phase 2 of the large-dataset monotonic-depth benchmark program
([program note](2026-07-11-large-dataset-benchmark-program.md)). This is a
placeholder capturing scope; it needs its own brainstorming → design → plan
cycle before implementation.

## Why separate

Learning-to-rank does not fit the current row-independent CSV + IQM harness. It
needs new protocol machinery, which is exactly why it is its own phase.

## Sketch

- **Dataset.** MSLR-WEB30K: ~3,771,125 query–document pairs, 31,531 queries, 136
  features, 5-level relevance. The canonical large **monotone-feature** LTR
  benchmark (BM25 / term-frequency / PageRank features carry accepted monotone
  priors; monotone constraints are standard in LightGBM here). Microsoft
  redistribution terms to be confirmed for hosting class. (Yahoo LTR a possible
  second curve.)
- **New machinery.** A query-grouped loader; a ranking objective; **NDCG@k** as
  the metric; **group-aware CV** (split by query, never leak documents of a query
  across folds); a ranking head/aggregation compatible with the monotone layers.
- **Screen + gate.** Same deep/shallow max-size screen and gate as Phase 1, with
  the margin expressed on NDCG.

## Open questions (for its brainstorm)

- Which monotone feature subset, and the direction justification per feature.
- Pointwise vs pairwise vs listwise objective, and how monotonicity composes with
  the ranking head.
- MSLR redistribution terms → LFS vs script-only.
