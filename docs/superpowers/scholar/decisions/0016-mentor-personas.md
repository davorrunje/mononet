# ADR-0016: Mentor personas author-selectable; no personality inference

- Status: accepted · Date: 2026-07-17 · Deciders: Davor Runje

## Context

Not every mentor/reviewer voice suits every author. The question was how to pick a
persona for the grill/advise voice and whether to match it to the author's
personality. A research pass separated grounded practice from folklore.

## Decision drivers

- Style typologies are solid; **trait-matching is a myth** (learning-styles
  debunk; Big-Five "fit" weak).
- AI personality inference is unreliable and ethically fraught, and would violate
  agency (profiling without consent).
- Autonomy support and calibrated challenge are the robust constants.

## Considered options

1. **A small set of author-*selectable* personas (sounding board / critical
   examiner / directive editor / opt-in devil's advocate), chosen by self-
   selection + task/stage suggestion + feedback calibration; never inferred.**
2. System infers personality and assigns a matched persona.
3. A single fixed voice.

## Decision

Option 1. Personas derived from supervision typologies (Lee × Gatfield), not
personality theory. Autonomy support constant; only directiveness/challenge vary.
Feedback targets the argument, not the person. Rule: *match the voice to the task
and the author's stated choice, never to an inferred personality.*

## Consequences

- Useful variety without pseudo-science or hidden profiling.
- Task/stage suggestions are defaults the author overrides.

## Rejected alternatives

- **Inferred matching** — unsupported (learning-styles myth) + consent/agency
  violation + unreliable inference.
- **Single fixed voice** — ignores real task/stage differences.

## Links

meta-spec §3.7; sub-spec 1 §6; digest `mentor-personas.md`.
