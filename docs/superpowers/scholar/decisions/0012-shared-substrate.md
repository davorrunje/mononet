# ADR-0012: Shared asset substrate — share the mechanism, not the file

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

`literature` and `dataset` both manage externally-sourced, identified, licensed,
mirror-able assets. Are their registries the same? A field-level comparison found
a modest shared provenance spine but larger, non-overlapping distinctive fields —
and literature has a standard format datasets lack (ADR-0008).

## Decision drivers

- Avoid a redundant second mirror/fixity/ID implementation.
- Don't force two genuinely different registries into one file.

## Considered options

1. **Share the *mechanism* (asset spine + rclone mirror + fixity + persistent-ID),
   keep two distinct registry *files/formats*.**
2. Fully unify into one registry.
3. Fully independent literature and dataset stacks.

## Decision

Option 1. One base asset record + one materialization/mirror engine in the
substrate; literature (bib+triage) and dataset (`datasets.yml`) extend it.

## Consequences

- No duplicated mirror logic; each front-end keeps its natural format.
- A base-record schema both must conform to.

## Rejected alternatives

- **Full unify** — false equivalence; loses the bib ecosystem.
- **Fully independent** — duplicates the non-trivial mirror/fixity/ID machinery.

## Links

sub-spec 4 §2; meta-spec §3.4.
