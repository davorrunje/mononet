# ADR-0019: Public plugin, named `scholar`

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

ADR-0001 chose a standalone plugin repo but left the name and visibility open. The
audience is the author's own repos, company colleagues, PhD peers — and, the
author now decides, the broader research community.

## Decision drivers

- Reach: share the methodology as widely as possible.
- The plugin is already domain-neutral by design (no ML/monotonic assumptions).
- Named counterpart to `superpowers` (engineering) — `scholar` (science).

## Considered options

1. **Public repo, named `scholar`.**
2. Private, shared only with colleagues/peers.
3. Decide visibility later.

## Decision

Option 1. Public, `scholar`.

## Consequences

- The plugin must be genuinely domain-neutral and self-documenting for strangers
  (drives the detailed root `README.md`).
- The plugin is licensed **Apache-2.0** (ADR-0022).
- No mononet/PhD-specific assumptions may leak into the plugin.

## Rejected alternatives

- **Private** — limits reach; the author explicitly wants broad sharing.
- **Decide later** — the audience is already known; deferring adds no value.

## Links

meta-spec §7 (Distribution), §10; ADR-0001.
