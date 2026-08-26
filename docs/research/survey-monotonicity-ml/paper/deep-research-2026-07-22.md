# Deep research — `survey-monotonicity-ml`

*`deep-research` workflow (web fan-out → fetch → 3-vote adversarial verification,
2/3-refute-to-kill → cited synthesis). Run 2026-07-22: 107 agents, 24 verified
claims merged to 12 findings, 0 errors. Complements the citation-graph scout
(`scout-2026-07-22.md` / `-sweep.md`) with web sources OpenAlex's forward-graph
misses — notably formal-methods/PL venues (FMICS, OOPSLA). Surfaces leads only;
inclusion/role are the author's triage calls.*

## Synthesis

Across the surveyed families, the 2023-2026 literature confirms rapid movement in every
direction: constrained architectures now have universal-approximation guarantees for
unbounded/alternating-saturation activations (Sartor 2025) and smooth min-max variants
that fix the classic zero-gradient pathology (Igel 2024); KAN-based monotonicity has
appeared (MonoKAN, and its use inside CKAN constitutive models);
certification/verification has matured from single MILP checks (Liu 2020; Vidot 2022) and
SMT counterexample loops (COMET 2020) to relational abstract-interpretation verifiers that
beat MILP by ~12x (RaVeN, OOPSLA 2024); soft/penalty methods improved where the penalty is
applied (Monteiro 2022 mixup penalties); and classical shape-constrained regression
persists (Smoothed Monotonic Regression 2017, monotone cubic B-splines 2023). Seminal
older anchors include UMNN (2019, integrate a positive derivative), Sill min-max networks,
and Lipschitz-residual constructions (Kitouni 2023) now shipped in tooling (MATLAB
constrained-deep-learning). All 24 surviving claims are backed by primary sources; the
main gaps a citation-graph search would miss are the formal-methods/PL-venue verifiers
(FMICS, OOPSLA) and vendor/tooling implementations.

## Findings (cross-checked against the citation-scout leads)

🆕 = not surfaced by the citation-graph scout; ↺ = already an anchor/scout lead.

| # | status | paper | family | conf. | source |
|---|---|---|---|---|---|
| 1 | 🆕 **NEW** | Igel, 'Smooth Min-Max Monotonic Networks' (ICML 2024) replaces the hard min/max of classical Sill/MM min-max… | constrained neural… | high | https://arxiv.org/abs/2306.01147 |
| 2 | ↺ anchor/rival (registered) | Sartor et al., 'Advancing Constrained Monotonic Neural Networks / Achieving Universal Approximation Beyond Bounded… | constrained neural… | high | https://arxiv.org/abs/2505.02537 |
| 3 | ↺ seed-anchor (UMNN) | Wehenkel & Louppe, 'Unconstrained Monotonic Neural Networks' (UMNN, NeurIPS 2019) guarantees monotonicity by modeling… | constrained-by-construction… | high | https://arxiv.org/abs/1908.05164 |
| 4 | ↺ seed-anchor (Liu 2020) | Liu, Han, Zhang & Liu, 'Certified Monotonic Neural Networks' (NeurIPS 2020) certifies monotonicity of general… | certification/verification… | high | https://arxiv.org/abs/2011.10219 |
| 5 | ↺ certification lead | Sivaraman, Farnadi, Millstein & Van den Broeck, 'Counterexample-Guided Learning of Monotonic Neural Networks' (COMET,… | certification/verification +… | high | https://arxiv.org/abs/2006.08852 |
| 6 | 🆕 **NEW** | Vidot, Ducoffe, Gabreau, Ober & Ober, 'Formal Monotony Analysis of Neural Networks with Mixed Inputs: An Asset for… | certification/verification | high | https://link.springer.com/chapter/10.1007/978-3-031-15008-1_3 |
| 7 | ↺ sweep lead (RaVeN) | Banerjee, Xu & Singh, 'Input-Relational Verification of Deep Neural Networks' (RaVeN, PACMPL/OOPSLA 2024) is a… | certification/verification | high | https://ggndpsngh.github.io/files/raven.pdf |
| 8 | ↺ sweep lead (Kitouni) | Kitouni, Nolte & Williams, 'Expressive Monotonic Neural Networks' (arXiv:2307.07512, 2023; ICLR 2023) gives a… | constrained neural… | high | https://github.com/matlab-deep-learning/constrained-deep-learning/blob/main/documentation/AI-Verification-Monotonicity.md |
| 9 | 🆕 **NEW** | Monteiro et al., 'Monotonicity Regularization: Improved Penalties and Novel Applications...' (UAI 2022, PMLR v180)… | soft/penalty/regularization | high | https://proceedings.mlr.press/v180/monteiro22a/monteiro22a.pdf |
| 10 | ↺ first-pass scout lead | 'MonoKAN: Certified Monotonic Kolmogorov-Arnold Network' (arXiv:2409.11078, 2024) imposes (partial) monotonicity on a… | KAN-monotonic | high | https://www.sciencedirect.com/science/article/pii/S0022509625001887 |
| 11 | 🆕 **NEW** | Wang, Fan, Li & Liu, 'Monotone Cubic B-Splines with a Neural-Network Generator' (arXiv:2307.01748, 2023) fits… | isotonic/classical… | high | https://arxiv.org/abs/2307.01748 |
| 12 | 🆕 **NEW** | Burdakov & Sysoev, 'A Dual Active-Set Algorithm for Regularized Monotonic Regression' (J. Optim. Theory Appl. 2017)… | isotonic/classical… | high | https://link.springer.com/article/10.1007/s10957-017-1060-0 |


## Caveats

- One sub-claim was **refuted** (1–2): that convex-monotone activations with
  *non-positive* weights also achieve universality — do **not** cite as established.
- Two findings (Igel smooth-min-max; UMNN positioning) ran while the safety
  classifier was unavailable — both are well-known peer-reviewed papers; verify
  quotes against the PDFs before citing.
- Finding #8's source URL is vendor tooling; cite the paper (Kitouni et al.,
  arXiv:2307.07512 / ICLR 2023), not the repo.

## Full verified claims

### [1] (high, vote 3-0)

Igel, 'Smooth Min-Max Monotonic Networks' (ICML 2024) replaces the hard min/max of
classical Sill/MM min-max monotonic networks with strictly-increasing smooth
minimum/maximum functions to cure the zero-gradient training pathology while preserving
monotonicity, and provably inherits the MM architecture's asymptotic (universal)
approximation properties. Family: constrained neural architectures (min-max).

**Evidence:** Abstract states SMM uses 'strictly-increasing smooth minimum and maximum functions' and
'inherits the asymptotic approximation properties from the MM architecture.' Corollary 1 +
Theorem 1 formalize universality (Sill 1997; Daniels & Velikova 2010). Merges claims [0]
and [1].

- https://arxiv.org/abs/2306.01147

### [2] (high, vote 3-0)

Sartor et al., 'Advancing Constrained Monotonic Neural Networks / Achieving Universal
Approximation Beyond Bounded Activations' (ICML 2025) proves non-negative-weight MLPs with
activations that saturate on alternating sides are universal approximators of monotone
functions (generalizing CMNN beyond bounded activations to e.g. ReLU), and proposes an
'activation switch' f(x)=sigma(W+x+b)-sigma(W-x+b) that adjusts activation to weight sign,
removing weight reparameterization and improving init-robustness/training stability.
Family: constrained neural architectures (CMNN).

**Evidence:** Theorem 3.5, Prop 3.9 establish universality with alternating saturation across >=3-4
layers; Eq. 12 defines the sign-split activation switch. Explicit follow-up to Runje &
Shankaranarayana 2023. Merges claims [2] and [3]. Note: the sibling claim that convex-
monotone activations with non-positive weights also achieve universality was REFUTED
(1-2).

- https://arxiv.org/abs/2505.02537

### [3] (high, vote 3-0 (positioning [5]); 2-1 (mechanism/family [4]))

Wehenkel & Louppe, 'Unconstrained Monotonic Neural Networks' (UMNN, NeurIPS 2019)
guarantees monotonicity by modeling a strictly-positive derivative with a free-form
unconstrained network and integrating it, rather than constraining weights/activations; it
argues prior weight/activation-constrained architectures trade expressiveness for
invertibility and removes that cap. Family: constrained-by-construction architecture /
monotone (autoregressive) flow (UMNN-MAF). Seminal older anchor.

**Evidence:** Abstract: monotonicity holds 'as long as its derivative is strictly positive,' enforced by
'a free-form neural network whose only constraint is the positiveness of its output';
contrasts with weight/activation constraints that 'enable invertibility but lead to a cap
on the expressiveness.' Flagship application UMNN-MAF is a normalizing flow. Merges claims
[4] and [5].

- https://arxiv.org/abs/1908.05164

### [4] (high, vote 3-0 (2-1 on the ramp-up recipe detail [22]))

Liu, Han, Zhang & Liu, 'Certified Monotonic Neural Networks' (NeurIPS 2020) certifies
monotonicity of general piecewise-linear (ReLU) networks by transforming verification into
a MILP problem, requiring no human-designed weight-space constraints and supporting
arbitrary architectures; the training recipe trains an off-the-shelf net with heuristic
monotonicity regularization whose magnitude is ramped up until the MILP verifier certifies
it. Family: certification/verification (with training-time regularization).

**Evidence:** Abstract verbatim: 'certify the monotonicity of the general piece-wise linear neural
networks by solving a mixed integer linear programming problem... does not require human-
designed constraints on the weight space,' training by 'gradually increasing the
regularization magnitude until it passes the monotonicity verification.' Merges claims
[8], [9], [21], [22]. This is RaVeN's 'Liu et al. [48]' ReLU baseline.

- https://arxiv.org/abs/2011.10219
- https://proceedings.neurips.cc/paper/2020/file/b139aeda1c2914e3b579aafd3ceeb1bd-Paper.pdf

### [5] (high, vote 3-0)

Sivaraman, Farnadi, Millstein & Van den Broeck, 'Counterexample-Guided Learning of
Monotonic Neural Networks' (COMET, NeurIPS 2020) enforces monotonicity on general
unrestricted ReLU networks (no restriction of the hypothesis space), provides
provable/certified monotonicity guarantees at prediction time (a monotonic envelope), and
trains via a CEGIS-style loop that iteratively mines and incorporates monotonicity
counterexamples. Family: certification/verification + counterexample-guided training.

**Evidence:** Abstract: 'we target general ReLU neural networks and do not further restrict the
hypothesis space'; 'develop a counterexample-guided technique to provably enforce
monotonicity constraints at prediction time'; 'iteratively incorporating monotonicity
counterexamples in the learning process.' Merges claims [10], [11], [12]. Uses an SMT
solver (distinct from Liu et al.'s MILP).

- https://arxiv.org/abs/2006.08852

### [6] (high, vote 3-0)

Vidot, Ducoffe, Gabreau, Ober & Ober, 'Formal Monotony Analysis of Neural Networks with
Mixed Inputs: An Asset for Certification' (FMICS 2022, LNCS 13487) uses a MILP solver to
verify partial monotonicity of trained networks with mixed inputs, and instead of a binary
verdict quantifies violation via lower/upper bounds on the input-space volume where
monotonicity fails, a metric named 'Non-Monotonic Space Coverage'. Family:
certification/verification.

**Evidence:** Paper (DOI 10.1007/978-3-031-15008-1_3; open access hal-03855271) uses MILP to verify
monotonicity of mixed-input networks and 'provides a lower and upper bound of the space
volume where the property does not hold, denoted Non-Monotonic Space Coverage.' A formal-
methods (FMICS) venue a citation-graph/ML-only search would miss. Merges claims [13] and
[14].

- https://link.springer.com/chapter/10.1007/978-3-031-15008-1_3

### [7] (high, vote 3-0)

Banerjee, Xu & Singh, 'Input-Relational Verification of Deep Neural Networks' (RaVeN,
PACMPL/OOPSLA 2024) is a general input-relational DNN verifier that lists monotonicity as
a supported relational property (alongside universal adversarial perturbations, targeted
UAP, Hamming distance), encoding monotonicity over a pair of DNN executions; its DiffPoly
abstract domain verifies monotonicity directly without any MILP and ~12x faster than the
prior SOTA (Boston Housing: [96,95,95]/98 at eps=[10,20,30] in 0.02s vs 0.25s for the Liu
et al. ReLU baseline). Family: certification/verification.

**Evidence:** Paper (ACM DOI 10.1145/3656377): 'Monotonicity can be verified directly by DiffPoly
without the need for any MILP formulation'; Table 5 / Sec 5.5 gives 0.02s vs 0.25s and
[96,95,95] vs all-98 for Liu et al. [48]. A PL-venue (OOPSLA) result a citation-graph ML
search would miss. Merges claims [15] and [16].

- https://ggndpsngh.github.io/files/raven.pdf

### [8] (high, vote 3-0)

Kitouni, Nolte & Williams, 'Expressive Monotonic Neural Networks' (arXiv:2307.07512, 2023;
ICLR 2023) gives a weight-constrained + single-residual Lipschitz construction for exact
monotonic dependence on any input subset while bounding the Lipschitz constant; it is
implemented in MATLAB's constrained-deep-learning framework as f(x)=g(x)+lambda*sum_{k in
S} x_k (g Lipschitz-continuous), guaranteeing df/dx_k = dg/dx_k + lambda >= 0 for monotone
inputs. Family: constrained neural architectures (Lipschitz-residual) + tooling.

**Evidence:** MATLAB doc states its monotonic construction is 'based on [1] Kitouni et al., Expressive
Monotonic Neural Networks, arXiv:2307.07512' and gives f(x)=g(x)+lambda*sum x_k with
df/dx_k >= 0 by construction. Vendor tooling implementation a citation search would miss.
Merges claims [17] and [18].

- https://github.com/matlab-deep-learning/constrained-deep-learning/blob/main/documentation/AI-Verification-Monotonicity.md
- https://arxiv.org/abs/2307.07512

### [9] (high, vote 3-0)

Monteiro et al., 'Monotonicity Regularization: Improved Penalties and Novel
Applications...' (UAI 2022, PMLR v180) proposes a monotonicity-penalty method computing
the gradient penalty over mixup interpolations of data-data and noise-data pairs,
enforcing monotonicity over a much larger input-space volume than prior penalty methods;
it identifies that Gupta et al. (2019) enforce monotonicity only where training data lie
and Liu et al. (2020) uniform draws only near input-space boundaries in high dimensions.
Family: soft/penalty/regularization.

**Evidence:** Abstract: prior penalties are monotonic 'only in a small volume'; the method 'uses
mixtures of training instances and random points to populate the space and enforce the
penalty in a much larger region,' with an n-sphere volume argument that uniform draws
concentrate at the boundary in high dimensions. Merges claims [19] and [20]; 'Omega_mixup'
label unverified but immaterial.

- https://proceedings.mlr.press/v180/monteiro22a/monteiro22a.pdf

### [10] (high, vote 3-0)

'MonoKAN: Certified Monotonic Kolmogorov-Arnold Network' (arXiv:2409.11078, 2024) imposes
(partial) monotonicity on a KAN by constraining spline control points; it is applied
inside Constitutive KANs (CKAN, J. Mech. Phys. Solids / arXiv:2502.05682, 2025) so strain
energy increases monotonically with strain, improving robustness and extrapolation.
Family: KAN-monotonic.

**Evidence:** CKAN paper: 'By imposing (partial) monotonicity on B-splines, MonoKAN ensures that strain
energy increases with strain, enhancing both robustness and extrapolation... using the
MonoKAN architecture for partial monotonicity constraints.' MonoKAN enforces monotonicity
via cubic Hermite spline control-point constraints. Claim [7].

- https://www.sciencedirect.com/science/article/pii/S0022509625001887
- https://arxiv.org/abs/2409.11078

### [11] (high, vote 3-0)

Wang, Fan, Li & Liu, 'Monotone Cubic B-Splines with a Neural-Network Generator'
(arXiv:2307.01748, 2023) fits monotone curves by placing monotonicity constraints on cubic
B-spline coefficients (a NN only generates/approximates spline solutions) — shape-
constrained spline regression, not a monotone neural architecture. Family:
isotonic/classical shape-constrained.

**Evidence:** Abstract: monotone fitting 'is equivalent to putting a monotonicity constraint on the
coefficients' of cubic B-splines. Claim [6].

- https://arxiv.org/abs/2307.01748

### [12] (high, vote 3-0)

Burdakov & Sysoev, 'A Dual Active-Set Algorithm for Regularized Monotonic Regression' (J.
Optim. Theory Appl. 2017) proposes Smoothed Monotonic Regression (SMR), adding a quadratic
regularization term to the monotonicity-constrained least-distance problem to yield a
convex QP. Family: isotonic/classical shape-constrained (with smoothing regularization).
Seminal older anchor.

**Evidence:** Abstract: 'we introduce a regularization term... formulated as a least distance problem
with monotonicity constraints. The resulting Smoothed Monotonic Regression (SMR) is a
convex quadratic optimization problem.' Monotonicity remains a hard constraint; only
smoothing is soft. Claim [23].

- https://link.springer.com/article/10.1007/s10957-017-1060-0

## Next steps

- Merge with the citation-scout leads; dedup; the **new** ones (Igel 2024, Vidot
  2022 FMICS, Monteiro 2022 UAI, Wang 2023 B-splines, Burdakov & Sysoev 2017) are
  the highest-value additions.
- Author triage (`position --level paper`): assign role/disposition, then ingest
  survivors into `references.json` + `triage.yml` with sign-off.
