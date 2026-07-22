---
status:
  level: paper
  id: survey-monotonicity-ml
  verdict: null
  readiness: drafting
  signed-off-by: null
  signed-off-date: null
  evidence: []
  covers: []                    # a survey is thesis BACKGROUND, not a contribution aim;
                                # thesis wiring (background chapter + qualifying milestone)
                                # to be settled in thesis framing (#139)
  load-bearing: null
  understanding: {status: pending, unresolved: []}
  blockers:
    - S2 contexts pending — keyless completeness sweep done (see paper/scout-2026-07-22-sweep.md); it proved the arXiv-vs-published split does not exist in OpenAlex (CMNN/Sartor have no PMLR record, Cano anchor already published) so recovery was 0; closing the intrinsic OpenAlex under-coverage now needs an active S2_API_KEY (currently HTTP 403)
    - empirical scope depends on how many rival methods we re-implement in the harness
  last-updated: 2026-07-22
---

# Pitch: A survey of monotonicity methods in machine learning

*Seeded by `paper-exploration` (promote) from the portfolio backlog row
`survey-monotonicity-ml`, per the author's scope decisions (2026-07-22). To be
developed by `paper-synthesis`: pitch → positioning → outline → decision →
sections. The author authors and signs; the skill drafts.*

## Central claim

There is no current, comprehensive treatment of **guaranteed / encouraged
monotonicity in machine learning** spanning the deep-constrained-architecture era.
The last broad overview (Cano et al., *Monotonic classification: an overview*,
2019) predates min-max-successor deep constructions (CMNN 2023, Sartor 2025),
modern deep lattice networks, and certified-monotonicity methods. This survey
unifies the field under one **taxonomy organized by how the monotonicity guarantee
is obtained**, backs the comparison with an **original reproducible benchmark**
(re-evaluating representative methods on a shared protocol via the `mononet`
benchmark harness), and **synthesizes the universal-approximation theory** into a
single statement of what each construction can represent and under what conditions.

## Contribution (three pillars)

1. **Taxonomy + critical synthesis.** Method families by guarantee mechanism:
   - *Constrained architectures* — min-max networks, CMNN (`mixed`), Sartor
     (`split`), monotonic dense units.
   - *Lattice methods* — (deep) lattice networks, lattice regression.
   - *Certification / verification* — counterexample-guided training,
     MILP/SMT-certified monotonicity.
   - *Soft / regularization* — pointwise monotonicity penalties, PWL methods.
   - *Classical* — isotonic regression, monotone GAMs, monotone trees/GBMs.
   - *Monotone / invertible flows & injective primitives.*

   Cross-cut by: hard vs. soft guarantee, full vs. **partial** monotonicity,
   expressivity, training cost, and when-to-use. A comparison table is the spine.
2. **Reproducible empirical study.** Re-evaluate representatives from each family
   under one protocol on the shared, fixity-pinned dataset registry
   (`datasets.yml`; 10 datasets) using the `benchmarks/` harness — plus original
   cross-method ablations. This is the survey's differentiator and reuses existing
   `mononet` infrastructure.
3. **Approximation-theory synthesis.** Which constructions are universal
   approximators of monotone functions, and under what activation / architectural
   conditions (Daniels & Velikova; the min-max universality line; Sartor's
   bounded-activation result). One framework, one statement per family.

## Target venue + bar

**TMLR** (rigorous, no page limit, survey-friendly, DOI-citable). Bar to clear:
- a **stated, systematic search methodology** (inclusion/exclusion) so coverage is
  defensible;
- an organizing taxonomy that does real work (not a list);
- critical synthesis + comparison table;
- the reproducible benchmark with released code/configs;
- the theory synthesis;
- an open-problems section.

## Relationship to the thesis

Serves double duty (author decision 2026-07-22): a **standalone TMLR paper** *and*
the thesis's **background / related-work foundation** (feeds the kappa's
"independent related work"), recorded as the **PhD qualifying-exam milestone**.
It does not cover a contribution aim (aim-1/2/3); its thesis role is background.
Concrete wiring into `thesis/aims.md` + `thesis/milestones.yml` is part of the
thesis-framing discussion (#139).

## Load-bearing hypotheses (to promote via `hypothesis-exploration`)

The empirical pillar's claims become testable hypotheses under the harness — e.g.:
- Under a unified protocol, hard-constrained architectures match or beat soft /
  regularization methods on the shared datasets (accuracy at equal guarantee).
- Partial-monotonicity support and cost separate the families in practice, not
  just in principle.

_(Seed rows are in this paper's `backlog.md`; promote as the design firms up.)_

## Immediate next steps (paper-synthesis)

1. ✅ **First-pass `literature scout --level paper`** run 2026-07-22 (keyless
   OpenAlex) — [`paper/scout-2026-07-22.md`](scout-2026-07-22.md). 243 citing works
   across 5 anchors; surfaced the method families + new-since-2023 leads (MonoKAN,
   MoST, MCNet, isotonic-embedding, counterexample/positivity certification).
2. ✅ **Completeness sweep (keyless)** run 2026-07-22 —
   [`paper/scout-2026-07-22-sweep.md`](scout-2026-07-22-sweep.md). Verified anchor
   records: CMNN/Sartor have **no** separate PMLR record in OpenAlex and Cano's
   anchor was already the published Neurocomputing record, so the arXiv-vs-published
   split does not exist here — published-record recovery was **0**; corpus stays 243.
   The low counts are intrinsic OpenAlex under-coverage. Added 7 method leads
   (incl. Lipschitz provably-monotonic nets, monotone-net approximation theory,
   MonoNet). **S2 contexts still pending** (S2 key returns HTTP 403 — inactive).
3. **S2 activation (still pending):** obtain/activate `S2_API_KEY`
   (`honest-scholar keys set S2_API_KEY`) then re-run for SciCite intents/contexts
   and a fuller citation graph — the only real fix for the CMNN/Cano/Sartor
   under-coverage the sweep proved is intrinsic to keyless OpenAlex. Manually seed
   known method anchors OpenAlex misses (min-max/Sill, UMNN/Wehenkel-Louppe).
4. Fix the empirical scope: which rival methods get re-implemented in the harness
   (drives feasibility).
5. `position --level paper` → move surviving leads into `references.json` +
   `triage.yml` (first-pass 14 + sweep 7), then develop positioning → outline.
