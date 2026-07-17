# ADR-0013: Experiment-backend *contract* + pluggable backend

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The pipeline skills need experimental evidence, but how experiments run is
project-specific (mononet's GPU fan-out means nothing to a colleague's project).
The author wanted the testing layer decoupled and hot-swappable.

## Decision drivers

- Pipeline skills must not depend on a concrete runner.
- mononet will adopt its own benchmark orchestration (PR #127) as the backend.
- Evidence must be citable and reproducible (provenance).

## Considered options

1. **A 4-capability contract (`run` / `evidence` / `tables` / `is-current`),
   bound per project; default `mononet-bench`.**
2. Bake mononet's orchestration directly into the skills.
3. Depend on an external tracker (MLflow/W&B).

## Decision

Option 1. The plugin ships the contract; each repo supplies an implementation.
Docs cite opaque **run-refs**; the provenance stamp carries dataset
`id+version+sha256`; the backend stamps evidence but never adjudicates it.

## Consequences

- Any project can bring its own runner; PINN (PR #116) migrates to the default.
- run-ref format left opaque (open item).

## Rejected alternatives

- **Bake in mononet's orchestration** — non-reusable; couples skills to one tool.
- **External tracker** — not git-native; heavy; provenance not repo-owned.

## Links

sub-spec 4 §3; orchestration spec (PR #127).
