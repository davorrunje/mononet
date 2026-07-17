# ADR-0010: Tiers A/B/C by (license × access); mirror ≠ redistribution grant

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Datasets differ in whether they can be committed, auto-fetched, or only manually
acquired. The author wants a private copy of every dataset regardless. A standards
pass grounded the policy (FAIR, licensing).

## Decision drivers

- Redistribution rights (may we republish bytes?) and access automation (can a
  machine fetch?) are orthogonal and both matter.
- FAIR A1.2/A2: metadata stays open even when bytes are gated.
- Legal correctness: a private mirror does not confer redistribution rights.

## Considered options

1. **Three tiers — A committed (small + redistributable), B auto-retrievable, C
   manual/gated — with a per-project private mirror across all tiers.**
2. Two tiers (in-repo vs external).
3. No tiering; commit everything or fetch everything.

## Decision

Option 1. Tier set by license × access; tier proposed by the skill but confirmed
by the human. **Hard rule:** tier/`redistributable` are set by the license, never
by mirror presence; never promote to Tier A because a mirror exists.

## Consequences

- Reproducibility (Tier A/B) + legal safety (Tier C) + durability (mirror).
- Tier assignment is a human-confirmed judgement.

## Rejected alternatives

- **Two tiers** — conflates the license and access axes.
- **No tiering** — either bloats the repo or breaks on gated/large/non-redist data.

## Links

sub-spec 3 §3; digest `dataset-management-standards.md`.
