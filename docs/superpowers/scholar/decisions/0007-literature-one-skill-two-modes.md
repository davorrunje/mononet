# ADR-0007: `literature` = one skill, scout/position modes

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Literature work spans generative idea-scouting (mining forward citations) and
defensive positioning (related-works synthesis for a committed claim/paper). Two
research passes examined whether these are one method or two, and whether they
split by level.

## Decision drivers

- Both rest on one citation-graph substrate (OpenAlex + Semantic Scholar).
- Within each function, hypothesis vs paper level is a *parameter* (ranking,
  depth, stopping), not a skill boundary.

## Considered options

1. **One `literature` skill, two modes (`scout`, `position`), each level-
   parameterized.**
2. Two separate skills (`lit-scout`, `lit-position`).
3. Per-level skills (hypothesis-literature, paper-literature).

## Decision

Option 1. Shared graph plumbing; modes differ by intent/output; a `level`
parameter tunes each.

## Consequences

- One methodology home; no duplicated graph plumbing.
- Modes are distinct front-ends; the shared substrate is the backend.

## Rejected alternatives

- **Two skills** — duplicates the substrate for a cosmetic split.
- **Per-level skills** — splits one body of methodology for form, not substance.

## Links

sub-spec 2 §1–§3; digests `citation-scouting.md`, `related-works-synthesis.md`.
