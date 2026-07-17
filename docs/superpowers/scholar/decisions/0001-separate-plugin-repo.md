# ADR-0001: Package the workflow as a standalone plugin, not in-mononet

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The research-workflow skills were being designed inside `mononet`, but the
workflow is domain-neutral and the author wants to share the *same* methodology
across their own repos, company colleagues, and PhD peers — reducing shared
cognitive load, not just tidying one repo.

## Decision drivers

- Reuse across repos and people; a single, uniform way of working.
- Decoupling from mononet internals (already begun via the backend contract).
- Peers must be able to install it against their own repos.

## Considered options

1. **Standalone Claude Code plugin repo** (`scholar`), mononet as one consumer.
2. Keep skills in `mononet/docs/superpowers/` (in-repo only).
3. A shared internal snippet library copied per repo.

## Decision

Option 1. `scholar` is a standalone, domain-neutral plugin; `mononet` becomes the
first *consumer*.

## Consequences

- Forces a clean plugin↔consumer boundary and contract-based decoupling (good).
- Requires migrating PR #128's in-repo specs to the plugin repo (ADR references
  meta-spec §9).
- Adds a repo to maintain; naming/visibility deferred (meta-spec §10).

## Rejected alternatives

- **In-repo only** — cannot be shared with peers/colleagues or reused; defeats the
  primary goal.
- **Copy-paste snippet library** — no versioning, drifts per repo, no uniformity.

## Links

meta-spec §1, §7, §9.
