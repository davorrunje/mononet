# Structure-Preserving PINNs (Paper 1)

Hard, expressive monotonicity (via `mononet`) as a PDE **admissibility prior**:
a Physics-Informed Neural Network whose architecture guarantees a
Total-Variation-Diminishing, entropy-admissible, oscillation-free solution **by
construction**, where soft-penalty PINNs only approximate it.

- **Forward mechanism tier:** Burgers-Riemann, Burgers smooth→shock, linear
  advection, LWR — validated against exact / TVD finite-volume ground truth.
- **Deep flagship (inverse):** traffic state estimation — reconstruct a
  monotone-front density field from sparse, noisy observations.

Scope: the **monotone-solution class** (single-front / queue / Riemann
scenarios). Abstract and headline result land here once experiments run
(manuscript in [`paper/paper.md`](paper/paper.md); reproduction in
[`RUNBOOK.md`](RUNBOOK.md)).
