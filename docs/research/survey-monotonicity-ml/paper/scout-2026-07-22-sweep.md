# Literature scout — completeness sweep — `survey-monotonicity-ml`

*honest-scholar `literature`, mode=scout, level=paper. Sweep run 2026-07-22,
**keyless OpenAlex** (polite pool, `mailto=davor.runje@fer.hr`). This is a
**delta / superseding sweep** over the first pass
[`scout-2026-07-22.md`](scout-2026-07-22.md) — read that first for method,
taxonomy, and the full 243-work family tables. This file records what the
completeness sweep changed (and, importantly, what it did **not**).*

> **Semantic Scholar unavailable (HTTP 403).** An S2 key exists but is currently
> inactive, so any S2-backed field (SciCite intents, citation-context snippets) is
> empty by design — expected, not a bug. Ranking still uses recency × multi-anchor
> coupling × impact, not Method-vs-Background intent. S2 activation remains the real
> lever for the remaining recall/precision gap (see Next steps).

## Headline finding: the arXiv-vs-published split does **not** exist in OpenAlex

The first pass flagged a recall gap: CMNN, Sartor, and Cano were thought to be
resolved to *preprint* DOIs whose forward-cites were split from a higher-cited
*published* record. The sweep tested that hypothesis by title-searching OpenAlex for
every record matching each anchor's title and comparing `cited_by_count`. Result:

- **CMNN and Sartor have no separate published (PMLR/ICML) record in OpenAlex.** A
  quoted-title search (`title.search:"Constrained Monotonic Neural Networks"`)
  returns exactly one record with that title — the arXiv preprint. PMLR ICML volumes
  (v202 for 2023, v267 for 2025) are **not** separately indexed as citable works
  here, so there is nothing to merge. The arXiv records **are** the canonical
  OpenAlex handles.
- **Cano's anchor was already the published Neurocomputing record**, not a preprint.
  W2901948247 carries DOI `10.1016/j.neucom.2019.02.024` (Neurocomputing). A
  separate arXiv preprint exists (W4289285734, `arXiv:1811.07155`) but has **0**
  forward-cites in OpenAlex, so it adds nothing.

The implausibly low forward-cite counts are therefore **not** an arXiv/published
split — they are genuine **OpenAlex citation-graph under-coverage** for these three
works (Google Scholar shows CMNN and Cano each with 100+ citations; OpenAlex holds
6 and 6). Keyless OpenAlex cannot close this; only S2 (or manual seeding of known
method papers) can.

## Corrected anchor table (arXiv vs published, cited_by_count)

Counts verified two ways: the `cited_by_count` metadata field **and** a direct
`filter=cites:<id>` count — they agree.

| anchor | arXiv record | published record | forward-cites (arXiv → published) | recovery |
|---|---|---|---|---|
| CMNN (Runje & Shankaranarayana 2023) | W4281571206 | *none in OpenAlex* | 6 → 6 | 0 |
| Sartor 2025 (Advancing CMNN) | W4415031780 | *none in OpenAlex* | 1 → 1 | 0 |
| Cano 2019 (Monotonic classification: overview) | W4289285734 (0 cites) | **W2901948247** (Neurocomputing) | 6 (published; arXiv adds 0) | 0 |
| Daniels & Velikova 2010 | — | W2125406789 | 201 | (kept) |
| Deep Lattice Networks (You et al. 2017) | — | W2751607718 | 57 | (kept) |

**Anchor-id corrections vs the first pass:** none required. W2901948247 is confirmed
the correct high-cite published Cano record (there is no higher-cite duplicate). The
first-pass note that Cano/CMNN/Sartor "resolved to arXiv DOIs" was accurate only for
CMNN and Sartor; for Cano the anchor was already the published DOI. No anchor id
changed; the arXiv Cano preprint (W4289285734) was added to the union and contributed
0 new works.

## Corpus stats vs the first pass

Union of forward-citations across the corrected anchor set (CMNN arXiv, Sartor arXiv,
Cano published + Cano arXiv, Daniels, DLN), deduped by OpenAlex id:

- **243 unique** forward-citing works; **28** cited by >1 anchor — **identical to the
  first pass.** Adding the published/arXiv variants recovered **zero** new works.
- Per-anchor contributions: daniels 201 · dln 57 · cmnn 6 · cano 6 · sartor 1.
- **No cap applied.** Daniels (the largest) has 201 forward-cites, under the 400 cap;
  nothing was truncated.

The completeness sweep therefore confirms the first pass was **already complete with
respect to keyless OpenAlex**. The corpus does not grow from anchor-record fixes.

### Co-citation / bibliographic-coupling probe (null result, documented)

To squeeze extra recall from the sparse anchors, the sweep ran
`literature neighbors` (co-citation + bibliographic-coupling) on CMNN (W4281571206)
and Sartor (W4415031780), top 25 each, then enriched the 28 out-of-corpus neighbor
ids. The CMNN co-citation neighborhood is **dominated by a single structural-
engineering application** that cites CMNN (Fe-SMA confined-concrete-column work),
which drags in dozens of concrete/seismic-retrofit references (e.g. "Theoretical
Stress-Strain Model for Confined Concrete", cby 8342). **None** of the 28 neighbors
are monotonicity-*method* contributions. Sartor's coupling set is empty (its
references are not indexed). Conclusion: neighbors add no method leads here; the
signal is drowned by one high-degree application citer.

## Family breakdown (title-only classification of the 243)

Keyword classification over **titles only** — abstracts are null under keyless
OpenAlex, so these counts differ from the first pass's title+abstract counts and are
weaker; treat as indicative, not authoritative. Families are non-exclusive.

| family | count |
|---|---|
| constrained-architecture (weight-constraint / min-max / monotonic-dense) | 18 |
| soft / penalty / regularization | 18 |
| certification / verification | 14 |
| isotonic & classical (isotonic regression, monotone GAM/GBM/trees) | 8 |
| lattice | 7 |
| monotone / invertible flows & injective | 4 |
| interpretability (cross-cutting, not a guarantee mechanism) | 19 |
| unclassified (mostly application-domain adoptions) | 165 |

The distribution matches the first pass's shape: a long tail of application-domain
adoptions, with genuine method contributions concentrated in constrained-arch,
soft/penalty, and certification.

## Newly-surfaced method-contribution leads (sweep additions)

These are genuine **method** papers in the 243-corpus that the first pass's hand-pick
did **not** table (it listed some only inside the unfiltered family tables). Surfaced
by re-scanning the corpus for method-y, recent titles. Provenance mandatory:
`via` = surfacing anchor; OpenAlex id is the stable handle.

| lead | year | cby | via | OpenAlex | family / why it matters |
|---|---|---|---|---|---|
| Robust and provably monotonic networks | 2023 | 8 | dln | W3216653328 | certification + constrained-arch: Lipschitz-based provably-monotonic nets (Nolte et al.) — a canonical construction the first pass under-ranked |
| Size and Depth of Monotone Neural Networks: Interpolation and Approximation | 2024 | 2 | daniels | W4394994853 | theory: expressivity / approximation bounds for monotone nets — feeds the survey's approximation-theory pillar |
| MonoNet: enhancing interpretability in neural networks via monotonic features | 2023 | 17 | daniels | W4321611068 | constrained-arch + interpretability: monotonic-feature bottleneck |
| Certified Logic-Based Explainable AI – The Case of Monotonic Classifiers | 2023 | 6 | daniels | W4384913338 | certification: formal/logic-based guarantees for monotonic classifiers |
| Input-Relational Verification of Deep Neural Networks | 2024 | 9 | daniels | W4399872358 | certification / verification: relational property verification (monotonicity as an input-relational property) |
| Graph-enhanced and monotonic embeddings for tabular data representation | 2025 | 1 | daniels | W4416303460 | constrained-arch: monotonic embeddings for tabular models |
| Univariate Probability Density Estimation With Partially Monotone Neural Networks | 2025 | 0 | daniels | W4417070475 | constrained-arch: partial-monotone nets for density estimation (touches flows/CDF line) |

These are **in addition to** the 14 leads already tabled in the first pass
(Counterexample-trained W4206574431, Deep Isotonic Embedding W4389778576, PenDer
W3175895431, MonoKAN W4416410340, MoST W7161738941, MCNet W4410088776, ICEnet
W4376654493, Explainable-monotonic W4404452603, Lattice-LDA W4307875327,
Knowledge-intensive GBM W2998367160, Flexible-monotone W4389230198,
Positivity-certification W4415179136, REGLO W4393156835, Sensitivity-direction
W3190707218). The two lists together are the current method-lead pool.

## New-since-2023 highlights

Postdating CMNN 2023's related-work section — the survey's "what's new" spine.
Confirmed from the first pass: **MonoKAN** (certified-monotonic KAN, W4416410340),
**MoST** (verifiable monotone set transformer, W7161738941), **MCNet** (monotonic
calibration networks, W4410088776), **Deep Isotonic Embedding** (W4389778576), and
the **counterexample / positivity-certification** line (W4206574431, W4415179136).
Added this sweep:

- **Lipschitz provably-monotonic networks** — "Robust and provably monotonic
  networks" (W3216653328, 2023). A distinct guarantee mechanism (Lipschitz + residual
  connection) that belongs alongside CMNN/Sartor in the constrained-arch/certification
  discussion.
- **Monotone-net approximation theory** — "Size and Depth of Monotone Neural
  Networks" (W4394994853, 2024). Directly serves pillar 3 (approximation-theory
  synthesis).
- **Monotonic-feature interpretability** — MonoNet (W4321611068, 2023).
- **Logic-based certified monotonic classifiers** (W4384913338, 2023) and
  **input-relational verification** (W4399872358, 2024) extend the
  certification/verification family beyond the first pass.
- **Partially-monotone density estimation** (W4417070475, 2025) — a bridge to the
  flows/injective line, which remains thin (4 hits).

## Taxonomy implications

The pitched taxonomy-by-guarantee-mechanism holds and gains texture. Net additions
this sweep:

- **Certification / verification** is richer than the first pass suggested: Lipschitz
  provably-monotonic (W3216653328), logic-based certified classifiers (W4384913338),
  and input-relational verification (W4399872358) join REGLO, counterexample-training,
  positivity certification, and MoST. This family deserves its own subsection.
- **Constrained architectures** should explicitly cover the **Lipschitz** route
  (Nolte) as a sibling of the weight-constraint route (CMNN/Sartor) and the
  min-max/monotonic-dense route — three distinct hard-guarantee mechanisms.
- **Approximation theory** (pillar 3) now has an explicit anchor (W4394994853) beyond
  Daniels & Velikova and the min-max universality line.
- **Flows / injective** remains thin (4). Keyless OpenAlex forward-cites off the
  current anchors will not fill it — it needs family-specific anchors (own
  `injective-monotonic-flows` line) and/or S2.

## Remaining next steps

1. **Activate S2** (`honest-scholar keys set S2_API_KEY`) and re-run in this repo's
   `position --level paper` flow to obtain SciCite **intents/contexts**
   (Method-vs-Background) and, likely, a **larger citation graph** than keyless
   OpenAlex — this is the only real fix for the CMNN/Cano/Sartor under-coverage that
   the anchor-record sweep proved is intrinsic to OpenAlex.
2. **Manually seed known method anchors** that the OpenAlex forward-graph misses
   (min-max networks / Sill 1997; UMNN / Wehenkel & Louppe 2019; expressive-monotonic
   / Lipschitz line) rather than relying on forward-cites from CMNN, whose OpenAlex
   citation graph is only 6 works deep.
3. **Add family-specific anchors** where the graph is thin — especially flows/injective.
4. **Move surviving leads into `references.json` + `triage.yml`** (role / disposition /
   rationale) during `position --level paper`; combine the first-pass 14 leads with the
   7 sweep additions above as the starting triage set.

*Guardrail note: this sweep proposes and surfaces only. Nothing here is an
include/exclude adjudication — that is PRISMA work for `position` mode, author-signed.*
