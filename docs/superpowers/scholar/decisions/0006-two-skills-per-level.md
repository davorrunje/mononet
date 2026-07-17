# ADR-0006: Two pipeline skills per level (generate / resolve)

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

How to structure the lifecycle skills so hypothesis-level and paper-level are
uniform, and so the "missing" stages (related-works, publish-decision) fit
without proliferating skills.

## Decision drivers

- Object×action symmetry across levels.
- A firewall: explore proposes, resolve disposes, synthesize reports.
- The "missing" stages are analogs of existing staged docs, not new skills.

## Considered options

1. **Two skills per level: generate (exploration) + resolve (testing/synthesis)**,
   with staged docs; "related works" = positioning (dual of strategy), "publish
   decision" = decision (dual of findings).
2. One big lifecycle skill per level with sub-commands.
3. A skill per lifecycle stage (many small skills).

## Decision

Option 1. The two missing stages resolve into staged documents in the resolve
skill, preserving the mirror.

## Consequences

- Minimal skill count; strong symmetry.
- Related-works and publish-decision are documents/gates, not skills.

## Rejected alternatives

- **One big skill** — breaks the explore/dispose/report firewall; hard to reason
  about.
- **Skill-per-stage** — over-fragmented; duplicated context.

## Links

meta-spec §3.1–§3.2; sub-spec 1 §1–§2.
