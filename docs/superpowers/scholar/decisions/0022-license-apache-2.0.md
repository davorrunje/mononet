# ADR-0022: License the `scholar` plugin under Apache-2.0

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

ADR-0019 made the plugin public and flagged that a license must be chosen before
release. `mononet` is Apache-2.0.

## Decision drivers

- Permissive, widely trusted, patent-grant clause; broad adoption.
- Consistency with `mononet` (Apache-2.0) — one license across the author's work.

## Considered options

1. **Apache-2.0.**
2. MIT.
3. A copyleft license (e.g. MPL-2.0 / GPL).

## Decision

Option 1. Apache-2.0.

## Consequences

- A `LICENSE` (Apache-2.0) + copyright header convention ships in the plugin repo
  at creation.
- Permissive reuse by colleagues, peers, and the community; patent grant included.

## Rejected alternatives

- **MIT** — fine, but Apache-2.0's explicit patent grant + matching `mononet` win.
- **Copyleft** — unnecessary friction for a methodology/skills plugin meant for
  broad adoption.

## Links

meta-spec §7, §10; ADR-0019.
