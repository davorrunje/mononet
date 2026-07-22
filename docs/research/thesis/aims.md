---
status:
  level: thesis
  id: constrained-monotonic-networks
  verdict: n/a
  readiness: framing            # framing | synthesis | defensible
  signed-off-by: null           # defensibility sign-off lives in kappa.md
  signed-off-date: null
  evidence: []
  covers: []
  load-bearing: null
  understanding: {status: pending, unresolved: []}
  blockers: []
  last-updated: 2026-07-21
---

# Thesis aims & narrative

*Framing draft written by research-init (adopt). PROPOSED — the author decides
scope and signs it; refine, don't churn.*

## Aims

- **aim-1:** Give the constrained-monotonic-network construction a **correct,
  reproducible, multi-backend** realization (PyTorch / JAX / Keras) and reproduce
  the original CMNN results under a standard protocol.
- **aim-2:** **Characterize** the construction: when and why the flavors
  (`mixed` / `alternate` / `split`), initialization, and residual depth help or
  hurt — turning folklore into measured, ablated results.
- **aim-3:** **Extend** constrained monotonicity to new settings —
  structure-preserving physics-informed networks and injective-monotonic
  primitives / normalizing flows.

## Narrative through-line

Constrained monotonic networks are a principled way to bake domain knowledge
(monotonicity) into a model with guarantees. This thesis takes the construction
from **a single published result to a dependable, well-understood tool and then
outward into new applications**: first a faithful multi-backend implementation and
reproduction (aim-1), then a systematic characterization of the design space that
governs when it works (aim-2), then applications that exploit the guarantee in
domains beyond tabular monotone regression/classification (aim-3). The original
contribution is the move from *"a construction that works on some benchmarks"* to
*"a construction we understand, can reproduce anywhere, and can carry into new
structure-preserving settings."*

## Chapter ↔ paper map

| Aim | Supporting paper-ids | Covered? |
|---|---|---|
| aim-1 | `cmnn-multibackend` | yes (reproduction in progress) |
| aim-2 | `monotone-constructions` | yes (active thread) |
| aim-3 | `structure-preserving-pinns`, `injective-monotonic-flows` | partial (design only) |

<!-- Coverage — not paper count — is the binding norm. aim-3 is currently
     design-stage on both supporting papers; progress surfaces it as a gap. -->
