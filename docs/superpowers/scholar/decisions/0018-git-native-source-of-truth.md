# ADR-0018: Git-native plain text as source of truth; cross-repo out of scope

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Where does the authoritative state live — external services (Zotero DB, MLflow,
W&B) or git-tracked files? And should a thesis be able to aggregate papers across
separate repos?

## Decision drivers

- Reproducibility, diffability, PR-review, and shareability with peers without a
  shared account/service.
- Consistency with mononet's committed-provenance stance.
- Work-research and PhD thesis live in separate repos with separate lives.

## Considered options

1. **Git-tracked plain text (bib/YAML/markdown + committed provenance) is the
   source of truth; external tools are optional front-ends/caches.**
2. External trackers (Zotero DB, MLflow/W&B) as source of truth.
3. Support cross-repo thesis aggregation now.

## Decision

Option 1. Zotero is an optional bib front-end; Optuna/`.doit.db` and rebuilt
caches are not authoritative; registries/decisions/results are committed.
**Cross-repo aggregation is explicitly out of scope for now** (each repo's top
level is self-contained) — recorded as a future item / GitHub issue.

## Consequences

- Everything is reviewable and reproducible from the repo alone.
- A future cross-repo thesis roll-up would need a separate design.

## Rejected alternatives

- **External trackers as source of truth** — not git-native; heavy; provenance not
  repo-owned; needs shared accounts.
- **Cross-repo now** — out of scope; adds coupling before it's needed.

## Links

meta-spec §1 (non-goals), §5, §10.
