# ADR-0003: Agency principle — assistants, not researchers

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

An AI research-workflow tool could plausibly try to *do* research and make
scientific decisions. The author insisted the opposite: the skills assist, keep
accounts, advise, and discuss; the human makes every material decision and is
accountable. Grounded by a dedicated research pass.

## Decision drivers

- Research accountability is non-delegable and attaches only to named humans.
- Automation bias / over-reliance is a documented failure mode (esp. LLM
  citation fabrication).
- The author wants growth and ownership, not push-button papers.

## Considered options

1. **Assistant/advisor; every material decision is a human's, recorded with a
   named sign-off.**
2. Autonomous/semi-autonomous agent that proposes verdicts/decisions.

## Decision

Option 1, elevated to the plugin's highest guiding principle.

## Consequences

- Material decisions (`findings` verdict, publish `decision`, thesis
  defensibility) carry a named human sign-off; the backend stamps evidence but
  never adjudicates it.
- Drafts are proposals; the author authors. No unattended end-to-end generation.

## Rejected alternatives

- **Autonomous adjudication** — violates ICMJE/COPE authorship-accountability
  norms, EU AI Act Art. 14 human-oversight, and invites automation bias.

## Links

meta-spec ⚑ guiding principle + §2.1; digest `agency-principle.md`.
