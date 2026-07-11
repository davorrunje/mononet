# Program: large-dataset monotonic-depth benchmark

Status: Program note (living)

An expansion of Sub-project C (extended benchmarks). Motivation: PR #72 and the
loan size-ladder ([design](2026-07-10-loan-size-ladder-design.md)) leave "does
monotone depth pay off at scale?" resting on a single dataset. This program
assembles a roster of large, genuinely monotonic public datasets, screens depth
on each at max size, and routes each dataset by a fixed gate to a full ladder
study or the standard benchmark. Data is version-controlled (Git LFS now, Zenodo
later) where licensing permits.

The work decomposes into phases by **task type**, because each new task type
needs its own protocol machinery. Each phase gets its own spec → plan →
implementation cycle.

| Phase | Task type | Content | Spec |
|---|---|---|---|
| 1 | Tabular classification / regression | Roster of ~9 tabular datasets, LFS/script hosting, git-lfs in devcontainers, max-size deep/shallow screen → CI+floor gate → ladder-or-standard | [design](2026-07-11-large-dataset-screen-design.md) — **approved** |
| 2 | Learning-to-rank | MSLR-WEB30K (~3.8M pairs × 136 features, monotone LTR features), grouped loader, NDCG@k, group-aware CV, monotone screen | [stub](2026-07-11-ltr-monotonic-benchmark-design.md) — planned |
| 3 | Curve-regression | R&F-Inventory (monotone inventory estimation), multi-budget-point target, curve-reconstruction + monotonicity-violation metrics | [stub](2026-07-11-curve-regression-benchmark-design.md) — planned |

## Gate (shared across phases)

The max-size screen tunes both arms (deep `depth ∈ {6,10,16}` vs shallow
`depth ∈ [1,4]`, both `mode="absolute"`, `residual=True`) and reports
`Δ = IQM(deep) − IQM(shallow)` with a seed-bootstrap band. A dataset advances to
a full ladder study iff `delta_lo > 0` **and** the point `Δ` clears a practical
margin (0.005 accuracy / 1% RMSE / an NDCG margin for LTR); otherwise it folds
into the standard benchmark.

## Backlog (recorded, not yet specced)

- **Fannie/Freddie mortgage** — tens of millions of rows, textbook monotone
  credit features (FICO↓, LTV↑, DTI↑, note-rate↑ → risk↑). License forbids
  re-hosting the data or derived products → **manual-download README +
  script-only**, registration-gated, non-trivial ETL (join acquisition ↔ monthly
  performance, derive terminal-delinquency label). Would be the largest credit
  anchor. Deferred by owner decision (accept the collaborator friction later).
- **LHCb heavy-flavor trigger** — the canonical deployed monotonic classifier
  (Kitouni, Nolte & Williams; monotone in ∑pT, impact-parameter significance,
  flight distance). Data is **collaboration-internal / not public**; only
  synthetic toys are released. Follow-up: pursue a reconstruction or obtain a
  sample.
- **Synthetic physics-motivated monotonic datasets** — controllable ground-truth
  monotone generators (e.g. a physics-flavored displacement/momentum trigger
  toy) to stand in for the unavailable LHCb data. Owner intends to work on this
  later.
