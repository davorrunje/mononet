# ADR-0014: `progress` cross-cutting; semantic roll-up; anti-Goodhart

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

The author wants progress tracked at all levels (hypothesis/paper/thesis) with a
markdown dashboard. A research pass grounded how to do this without perverse
metrics.

## Decision drivers

- Avoid a separate progress artifact that drifts out of sync.
- Research is not decomposable/burndown-able; verdicts are binary-ish.
- A self-tracking tool is especially Goodhart-prone.

## Considered options

1. **A cross-cutting `progress` skill; status in each artifact's frontmatter;
   `dashboard` is a generated projection; semantic (coverage+blocker) roll-up;
   anti-Goodhart (surface gaps, never a score).**
2. A fourth "progress" level.
3. Percentage/velocity dashboards.

## Decision

Option 1. Status lives with the artifact (living-document); roll-up is a semantic
function of children + gate criteria; **refuted = done/green** (verdict and
readiness are distinct axes). Do not count words/papers/commits/success-rate.

## Consequences

- Nothing to game or keep in sync; honest coverage/blocker view.
- Roll-up rules must be specified per level (open item).

## Rejected alternatives

- **Fourth level** — progress isn't an object with a lifecycle.
- **Percentage/velocity** — assumes decomposable scope; Goodhart-prone.

## Links

meta-spec §3.6; sub-spec 1 §5; digest `thesis-and-progress-tracking.md`.
