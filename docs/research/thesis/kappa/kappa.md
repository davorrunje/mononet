---
status:
  level: thesis
  id: constrained-monotonic-networks
  verdict: n/a
  readiness: synthesis          # synthesis | defensible (defensible only after the gate + sign-off)
  signed-off-by: null           # REQUIRED for defensibility — named human
  signed-off-date: null
  evidence: []                  # the kappa introduces NO new findings
  covers: []
  load-bearing: null
  understanding: {status: pending, unresolved: []}
  blockers:
    - aim-3 uncovered by resolved results (structure-preserving-pinns and injective-monotonic-flows are design-only)
  last-updated: 2026-07-21
---

# Kappa: Constrained Monotonic Neural Networks — construction, characterization, applications

*Scaffold written by research-init (adopt). The kappa RECONTEXTUALIZES resolved
work; it introduces no new findings. The author authors; the skill drafts.*

## Introduction / background

<!-- The problem space: guaranteed monotonicity as a way to encode domain
     knowledge; the CMNN construction (Runje & Shankaranarayana 2023) as the
     starting point. -->

## Unifying narrative / aims

See [`../aims.md`](../aims.md) — construction (aim-1) → characterization (aim-2)
→ applications (aim-3).

## Independent related work

<!-- Thesis-scope related-works synthesis, broader than any single paper's
     positioning. Draw on `literature position`. Anchors: Runje 2023, Sartor 2025. -->

## Per-paper summary + contribution statement

### cmnn-multibackend
- **Summary:** *<multi-backend implementation + CMNN reproduction>*
- **Candidate's contribution:** *<author-written>*

### monotone-constructions
- **Summary:** *<flavors / init / depth characterization>*
- **Candidate's contribution:** *<author-written>*

### structure-preserving-pinns
- **Summary:** *<design-stage; application>*
- **Candidate's contribution:** *<author-written>*

### injective-monotonic-flows
- **Summary:** *<design-stage; application>*
- **Candidate's contribution:** *<author-written>*

## Unifying discussion

<!-- The original contribution to knowledge across all papers. -->

## Future work

<!-- Feeds paper-exploration. -->

---

## Defensibility gate

- **Aim coverage:** aim-1 ✓, aim-2 ✓, **aim-3 uncovered** (both supporting papers
  are design-only) — a blocker, surfaced honestly.
- **Through-line stated:** yes (see aims.md).
- **Original contribution stated:** draft (above).

### Mock viva + per-gap sign-off

<!-- ADR-0021 escalated gate: `defend` runs a full mock viva (claim | cited-work
     | methodology) before defensibility; the author acknowledges EACH gap in
     writing. Not defensible until signed-off-by + signed-off-date are set above. -->

| Surfaced gap | Acknowledged by | Date |
|---|---|---|
| aim-3 uncovered by resolved results | _(pending)_ | |
