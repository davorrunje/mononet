# ADR-0017: `research-init` one skill, init/adopt modes

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

A repo must be set up to use `scholar` — both fresh repos and existing ones with
prior research artifacts (mononet has benchmarks, reference PDFs, informal
hypotheses). This is the payoff for the "benchmarks folder out of control"
problem that motivated the effort.

## Decision drivers

- Both greenfield and brownfield onboarding are needed.
- They share the same end state (the consumer layout).
- Backfill is inventory-and-map on top of scaffolding.

## Considered options

1. **One `research-init` skill with `init` (greenfield) and `adopt` (backfill)
   modes.**
2. Two separate skills.

## Decision

Option 1. `adopt` inventories an existing repo and proposes mappings (PDFs/digests
→ bib+triage; benchmark data → `datasets.yml`; past results/specs → retroactive
hypotheses; orchestration → experiment backend), materialized with human
confirmation. mononet is the first `adopt`. Folded into the meta-spec (not its own
sub-spec).

## Consequences

- One onboarding entry point; brownfield is init + a map phase.
- Delegates per-item registration to the capability skills' verbs.

## Rejected alternatives

- **Two skills** — they drive to the same end state; splitting duplicates the
  scaffolding logic.

## Links

meta-spec §6.
