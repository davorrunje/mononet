# ADR-0002: Scientific scope only; engineering delegated to `superpowers`

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The full research lifecycle includes both scientific work (claims, evidence,
decisions, papers) and engineering (designing/building the code that produces
evidence). `superpowers` already handles engineering well (brainstorming →
writing-plans → implementation).

## Decision drivers

- Don't reinvent what `superpowers` does; keep `scholar`'s identity sharp.
- The scientific parts (rigor, positioning, decisions) are what's missing.

## Considered options

1. **Scientific scope only; delegate engineering to `superpowers`.**
2. One plugin covering both science and engineering.

## Decision

Option 1. `hypothesis-testing` hands design/plan/implementation to `superpowers`
and stores the artifacts under the hypothesis folder.

## Consequences

- Clean separation; `scholar` composes with `superpowers` rather than competing.
- A dependency on `superpowers` for the engineering leg (acceptable; both are
  plugins the user runs).

## Rejected alternatives

- **All-in-one plugin** — duplicates `superpowers`, bloats scope, blurs identity.

## Links

meta-spec §1 (non-goals), §2; sub-spec 1 §2.2.
