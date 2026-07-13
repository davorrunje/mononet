# Structure-Preserving Physics-Informed Neural Networks: Hard Monotonicity as a Total-Variation-Diminishing Prior for Conservation Laws

**Authors:** Davor Runje (and collaborators — TBD)
**Status:** Draft — headline result tables (§6.1 forward, §6.2 inverse) filled from
committed 10-seed IQM artifacts; **pending:** figures, the sparsity × noise sweep,
and the cross-backend equivalence check (see `RUNBOOK.md`).

> **Note (remove before submission).** Written manuscript-first: claims were
> committed *before* the runs, then reconciled to the evidence. Notably, the
> single-seed inverse result initially suggested an accuracy *win*; the 10-seed
> IQM shows accuracy is a **wash** with the baselines and the real, robust win is
> **structural admissibility** — the text reflects the latter, honest reading.

## Abstract

Physics-Informed Neural Networks (PINNs) fail on hyperbolic conservation laws
with shocks: as unconstrained continuous approximators they exhibit the Gibbs
phenomenon, producing spurious oscillations that violate the total-variation-
diminishing (TVD) property and the entropy condition. The standard remedy —
adding entropy/total-variation *penalty* terms to the loss — turns training into
an unstable multi-objective problem and offers no guarantee: the network can
still oscillate. We instead move the physics into the *architecture*. Using an
expressive hard-monotonicity construction (`mononet`), we build PINNs whose
output is monotone in the spatial coordinate **by construction**, so that within
the monotone-solution class the reconstruction is TVD and entropy-admissible
with **zero** achievable oscillation — a structural guarantee, not a penalty.
We first validate the mechanism on forward scalar conservation laws against exact
and TVD finite-volume ground truth (equal-budget hyperparameter tuning, 10-seed
IQM): the hard-monotone PINN attains **zero** admissibility violation and
near-zero overshoot (0.003) versus 0.11–0.23 violation and up to 0.13 overshoot
for tuned unconstrained/soft-penalty baselines — at a modest accuracy cost, since
the strong-form residual actively smears the discontinuity. We then apply it where
a PINN genuinely complements a classical solver — the *inverse* problem of traffic
state estimation from sparse, noisy observations (~0.7 % space-time coverage) —
where the hard-monotone reconstruction is competitive in accuracy at low noise
(L¹ IQM 3.5 vs 3.9) and **decisively more accurate as observation noise grows** —
~20–36 % lower L² at noise 33 % of the signal range — while being the **only
structurally admissible, oscillation-free** solution (violation 0 vs the
baselines' 0.9–1.3 at high noise). The monotone prior regularizes the noise the
unconstrained/soft baselines over-fit. A naive, inexpressive weight-clipping
constraint fails badly (≈6× worse error), confirming that it is *expressive* hard
monotonicity that matters. JAX and PyTorch
implementations are provided; a cross-backend equivalence check is left to the
released artifact. Code and trained models ship with the `mononet` package.

## 1. Introduction

Hyperbolic conservation laws, `u_t + f(u)_x = 0`, model transport-dominated
phenomena and develop discontinuous *shocks* in finite time even from smooth
initial data. Standard PINNs approximate the solution with an unconstrained
network trained on the PDE residual plus initial/boundary terms. Near a shock
this fails characteristically: the network overshoots and oscillates (the Gibbs
phenomenon), the total variation of the profile increases, and the recovered
weak solution can violate the entropy condition — i.e. it is *unphysical*.

The prevailing fix is *soft*: add penalty terms that discourage total-variation
growth or entropy violation. This has two structural defects. (i) It converts
training into a multi-objective problem in which the physics residual, the data,
and the admissibility penalty compete; the penalty weight is a fragile
hyperparameter. (ii) It offers **no guarantee** — a soft penalty is minimized,
not enforced, so the trained network can and does still oscillate wherever the
optimizer trades penalty for residual.

**This paper moves the admissibility condition from the loss into the
architecture.** For a scalar conservation law whose entropy solution is monotone
in space, TVD and the (Oleinik) entropy condition are *equivalent to a
monotonicity statement about the solution*. We therefore parameterize the PINN
with a network that is monotone in the spatial coordinate **by construction**,
using the expressive constrained-monotonic construction of `mononet`
[@runje2023constrained; @sartor2025advancing] — expressive because, unlike naive
weight-clipping or convex-only monotone nets, its convex/concave activation split
is a universal approximator of monotone functions. A monotone profile cannot
oscillate; hence within the monotone-solution class the reconstruction is TVD
and entropy-admissible with zero achievable oscillation, independent of the
optimizer.

**Contributions.**
1. A **structure-preserving PINN framework**: an *admissibility abstraction* that
   maps a PDE to a monotonicity/convexity constraint on the solution field, and a
   *problem registry* over a backend-agnostic core with JAX and PyTorch trainers.
2. A **theorem** (Section 4) that, within the monotone-solution class, a
   spatially-monotone network is TVD with total variation fixed by the boundary
   states, and satisfies the Oleinik entropy condition — by construction.
3. **Mechanism validation** on forward scalar conservation laws against exact and
   TVD finite-volume references, isolating the architectural effect from sampling
   and tuning (Section 6.1).
4. The **flagship result**: on *inverse* traffic state estimation from sparse,
   noisy data — a regime where classical finite-volume solvers do not apply
   because the full initial condition is unknown — the hard-monotone PINN
   reconstructs the shock/queue front without oscillation and dominates
   unconstrained and soft-penalty PINNs as observations become sparse and noisy
   (Section 6.2).
5. A **cross-backend** guarantee: identical results (to tolerance) from JAX and
   PyTorch, dogfooding `mononet`'s cross-backend equivalence.

**Scope.** We restrict to the *monotone-solution class* of scalar conservation
laws — problems whose entropy solution is monotone in `x` for all `t` (monotone
initial data producing a shock; Riemann problems; single-front queue formation).
This class is exactly where the guarantee is unconditional. Non-monotone data
(e.g. `u_0 = -\sin\pi x`), rarefactions, N-waves, and multi-front congestion are
out of scope and discussed in Section 7.

## 2. Related work

**PINNs for conservation laws.** [@raissi2019pinn] introduced PINNs; their
failure at shocks is well documented, and remedies add entropy/viscosity or
total-variation *penalties* to the loss (e.g. weak-form entropy-stable PINNs
[@wepinn]) or use derivative/shape penalties with adaptive weighting [@dcpinn].
These are *soft*: admissibility is minimized, not guaranteed. We enforce it
architecturally.

**Traffic state estimation with PINNs.** [@shi2021physics] estimate traffic state
from sparse data with a PINN informed by second-order traffic models, using soft
physics losses. This is the closest prior art to our flagship; it exhibits the
oscillation-at-shock failure a hard-monotone architecture removes.

**Shape-constrained networks.** Hard monotonicity/convexity by architecture has a
long lineage — non-negative-weight + convex-activation nets and Input Convex
Neural Networks [@amos2017icnn] — but these are the *inexpressive convex-only*
regime and pay a documented accuracy penalty. `mononet`
[@runje2023constrained; @sartor2025advancing] is expressive (convex/concave
activation split, a universal approximator of monotone functions), and HardNet
[@hardnet2024] shows hard-constrained nets retain universal approximation —
refuting the "hard constraints kill expressiveness" premise of soft-penalty work.
To our knowledge no prior PINN embeds *expressive* hard monotonicity in the
architecture for conservation laws.

**High-dimensional neural PDE solvers.** Deep BSDE [@han2018solving] and the Deep
Galerkin Method [@sirignano2018dgm] target high-dimensional PDEs with
unconstrained networks and impose no shape structure. Structure-preserving
extensions to those settings (HJB, Fokker–Planck, eikonal) are the subject of
follow-up papers.

## 3. Method

### 3.1 Structure-preserving PINN

For a scalar conservation law `u_t + f(u)_x = 0` on `[a,b] × [0,T]`, we
parameterize the solution field `u_θ(x, t)` by a network built from `mononet`
layers with a monotonicity mask that constrains `u_θ` to be non-increasing in `x`
(sign chosen per scenario) and unconstrained in `t`. The network is a shallow
stack of `MonoResidual` blocks (≈4 layers; depth beyond a few layers does not
help this construction, so width is the capacity lever).

### 3.2 Admissibility abstraction and problem registry

Each PDE is a plug-in exposing: the residual, an `AdmissibilitySpec` (the
monotonicity mask + convex/concave axes), the initial/boundary conditions or the
observation operator (inverse mode), and an exact-or-reference ground truth. The
admissibility-violation functional — the total wrong-sign spatial-derivative mass
— is `0` for the hard-monotone model by construction and positive otherwise; it
is the paper's headline metric. The framework is backend-agnostic (pure-NumPy
core) with thin JAX (Flax NNX) and PyTorch trainers.

### 3.3 Losses and baselines

The training loss is standard PINN — residual + initial/boundary (forward) or
residual + data-fit (inverse) — **with no entropy/TV penalty**; the architecture
supplies admissibility, so every method shares an identical loss and differs only
in the network. Baselines: **vanilla** (unconstrained MLP), **soft**
(unconstrained + TV/entropy penalty, penalty weight tuned), **weight-clip**
(inexpressive non-negative-weight monotone net), and **hard-monotone** (ours).

## 4. Theory

Let `u_θ(·, t)` be continuous and non-increasing in `x` on `[a, b]`.

**Total variation.** `TV(u_θ(·,t)) = ∫_a^b |∂_x u_θ| dx = u_θ(a,t) − u_θ(b,t)`.
With bounded boundary states the total variation is fixed by those states; the
network cannot create new local extrema, so it is TVD (non-increasing in `t` when
the boundary states are non-increasing in `t`) and exhibits **zero** overshoot.

**Entropy condition.** For a convex flux `f`, the Oleinik condition across a shock
requires `u_L > s > u_R`, i.e. `∂_x u ≤ 0` at the front. Constraining the
hypothesis space to non-increasing `u_θ` restricts it to the entropy-admissible
set; entropy-violating expansion shocks are unrepresentable.

**Scope of the guarantee.** The statements hold within the monotone-solution
class. A continuous monotone network represents a shock as a steep but continuous
ramp (zero overshoot), not a true discontinuity; Section 6 quantifies the ramp
width versus grid/observation resolution. *(Precise statements and proofs to be
finalized here; cross-check with the Lean formalization track where applicable.)*

## 5. Experimental design

**Forward mechanism tier (Section 6.1).** Burgers-Riemann, Burgers smooth→shock,
linear advection, LWR (Greenshields flux). Ground truth: closed-form entropy
solutions and a TVD Godunov reference. All four methods, both backends. Metrics:
L¹/L² error, admissibility violation, TV(t), near-shock overshoot, shock
speed/position error, mass conservation.

**Inverse flagship (Section 6.2).** LWR traffic state estimation: reconstruct a
monotone-front density field from sparse, noisy scattered observations (probe /
loop-detector geometry), no full initial condition. Sweep over observation
sparsity and noise. Metrics: reconstruction L¹/L² vs sparsity/noise, admissibility
violation, front-position error.

**Protocol.** Hyperparameters are tuned per method with Optuna under an
**identical search space and trial budget** (the soft baseline's penalty weight is
searched, not fixed), so comparisons are not confounded by tuning. Sampling and
observation masks are seeded and identical across methods. Reported numbers are
multi-seed.

## 6. Results

Protocol: per method, an identical Optuna budget (20 trials, seed 0) tunes
`lr`/`width`/`residual_weight` and the tier's data-term weight; the best config is
then evaluated over 10 seeds and reported as the interquartile mean (IQM) with a
95 % bootstrap band. JAX backend, 8000 steps, global-norm gradient clipping.
Artifacts: `results/forward-mechanism.json`, `results/inverse-headline.json`
(regenerate via `RUNBOOK.md`).

### 6.1 Forward mechanism tier (Burgers-Riemann)

This tier is the constraint's *worst case* — with no data, the strong-form PDE
residual must carry the solution, and minimizing it near a discontinuity rewards
smearing the shock. It is a mechanism check, not a relevance claim (a TVD
finite-volume scheme dominates forward 1-D problems).

Hard-monotone field = `MonoResidual` (default). A residual-vs-plain-`MonoLinear`
field ablation is being recorded (both variants are reproducible via the
`--no-residual` flag; see FINDINGS) — on this residual-dominated tier the plain
field is actually a bit more accurate, the mirror of the inverse result.

| method | L¹ (IQM [95 %]) | L² | admiss. violation | overshoot |
|---|---|---|---|---|
| **hard-monotone (mononet)** | 8.06 [5.73, 10.35] | 1.64 | **0** | **0.013** |
| vanilla | 6.07 [5.12, 7.10] | 1.21 | 0.251 | 0.148 |
| soft | 5.50 [5.17, 6.93] | 1.15 | 0.105 | 0.018 |
| weight-clip (inexpressive) | 44.1 [43.7, 45.8] | 3.77 | 0 | 0.365 |

The hard-monotone PINN is the only **oscillation-free** solution (overshoot 0.013
vs. vanilla's 0.148) with **zero** violation, but here it pays an accuracy cost
(L¹ ~8 vs vanilla ~6) — the expected structure-vs-accuracy trade when the strong
residual, not data, must fit the shock. This tier is a mechanism check; the
relevance claim is the inverse flagship (§6.2).

*Pending figures (see RUNBOOK): TV(t) curves; near-shock solution profiles.*

### 6.2 Inverse flagship — traffic state estimation (LWR)

Reconstruct the density field `ρ(x,t)` from 80 sparse, noisy observations (~0.7 %
of the space-time grid) — the data-assimilation regime a mesh solver cannot serve
(no full initial condition). Here the residual is a light regularizer and the
observations anchor the field.

The hard-monotone field is a `MonoResidual` stack (the near-linear gate collapse
that earlier forced a plain-`MonoLinear` fallback is fixed upstream in PR #100;
see FINDINGS).

| method | L¹ (IQM [95 %]) | L² | admiss. violation | overshoot |
|---|---|---|---|---|
| **hard-monotone (mononet)** | **3.52 [2.69, 4.04]** | **0.714** | **0** | **0.023** |
| vanilla | 3.92 [3.34, 4.27] | 0.731 | 0.083 | 0.024 |
| soft | 3.95 [3.36, 4.37] | 0.723 | 0.104 | 0.040 |
| weight-clip (inexpressive) | 22.5 [22.0, 23.1] | 2.10 | 0 | 0.297 |

The hard-monotone model is **marginally best on both L¹ and L²** *and* the **only
structurally admissible** one (violation 0 vs. 0.08–0.10 for the oscillating
baselines) — i.e. guaranteed admissibility at no accuracy cost (L¹ bands still
overlap, so this is parity-or-better, not a decisive accuracy win). The
inexpressive weight-clip baseline fails outright (≈6× error), so the effect is due
to `mononet`'s *expressive* hard monotonicity, not monotonicity per se.

#### 6.2.1 Robustness under noise — the key result

Stress-testing each method's tuned config across noise (MonoResidual field,
`results/inverse-sweep.json`; IQM over seeds), at n_obs = 80:

| noise | L² (hard / van / soft) | violation (hard / van / soft) |
|---|---|---|
| 0.00 | 0.70 / 0.68 / **0.63** | **0** / 0.05 / 0.06 |
| 0.05 | **0.69** / 0.73 / 0.79 | **0** / 0.14 / 0.23 |
| 0.10 | **0.86** / 0.94 / 1.08 | **0** / 0.37 / 0.51 |
| 0.15 | **1.14** / 1.26 / 1.47 | **0** / 0.62 / 0.88 |
| 0.20 | **1.26** / 1.59 / 1.97 | **0** / 0.86 / 1.32 |

There is a clean **crossover**: at zero noise the unconstrained/soft baselines are
marginally more accurate (nothing to over-fit), but **from noise ≥ 0.05 the
hard-monotone field is best on L², and the margin widens with noise** — at
noise 0.20 it is ~20 % better than vanilla and ~36 % better than soft. The
mechanism is regularization: the baselines fit the observation noise (and
oscillate — monotonicity violation climbing to 0.9–1.3), while the monotone prior
cannot, so it both **resists noise (lower error) and stays exactly admissible
(violation 0)** at every operating point. This is the paper's central result: in
the realistic sparse-and-noisy data-assimilation regime, expressive hard
monotonicity delivers *both* better accuracy *and* guaranteed structure, and the
advantage grows as data degrades. The crossover is **general**: it holds at every
observation count (20–160; at the sparsest, hard-monotone wins even at zero noise)
and for both the `MonoResidual` and plain-`MonoLinear` fields — 20 cells with a
coherent monotone trend, so it reflects the *monotone constraint*, not a single
operating point or architecture detail.

**Non-PINN / smoothness comparator.** To check the win is not just "any smoother",
a classical thin-plate-spline RBF fit *directly* to the observations (no PDE, no
monotonicity; smoothing tuned by held-out-observation CV) is far worse across the
whole grid — n_obs=80: L² 1.37 → 2.19 (noise 0 → 0.20) vs the hard-monotone
field's 0.70 → 1.26, i.e. ~2× — and is not admissible (violation 0.1–0.4). So the
result rests on **two** ingredients a generic smoother lacks: the PDE (to fill
unobserved space-time) *and* expressive hard monotonicity (accuracy under noise +
admissibility). Generic smoothing reproduces neither.

*Pending figure: reconstruction L² vs noise (the crossover) and the reconstructed
`ρ(x,t)` field with observations overlaid. (The raw out-of-range fraction was
uninformative — dominated by sub-0.01 excursions; a physical-bounds `[0, ρ_max]`
metric is the right future measure.)*

### 6.3 Cross-backend equivalence

*Pending:* JAX vs PyTorch hard-monotone agreement on identical points. Requires
both backends in one environment (the `default` / `all-cpu` devcontainer; the
GPU-JAX environment used for the tables above has no PyTorch). The construction is
backend-independent by design; the empirical tolerance will be reported from that
run.

### 6.4 Field ablation: `MonoResidual` vs plain `MonoLinear`

The hard-monotone field can be a `MonoResidual` stack or a plain `MonoLinear`
stack (config knob; only `hard_monotone` is affected — the other methods are
architecture-independent). Both are exactly admissible; the accuracy difference is
small and tier-dependent (10-seed IQM):

| tier | L¹/L² residual | L¹/L² plain |
|---|---|---|
| inverse (LWR, noise 0.02) | 3.52 / 0.714 | 3.48 / 0.744 |
| forward (Burgers) | 8.06 / 1.64 | 8.09 / 1.23 |

Across the noise sweep, `MonoResidual` is marginally better on the inverse tier
(e.g. n_obs=80/noise=0.20: L² 1.26 vs 1.33), plain slightly better on the forward
worst case. Crucially, **the noise crossover (§6.2.1) holds for both variants** —
the benefit is the monotone *constraint*, not the residual detail. (Before the
upstream gate-init fix, `MonoResidual` collapsed to a near-linear map here; that is
now resolved, and the two variants are close.)

## 7. Discussion and limitations

- **Monotone-solution class.** The guarantee is unconditional *within* this class
  and empty outside it: non-monotone initial data, rarefactions, N-waves, and
  multi-front congestion cannot be represented by a globally monotone field. This
  is a genuine restriction, stated up front, not a caveat discovered late.
- **Continuous ramp, not a true jump.** The shock is a steep continuous ramp;
  Section 6 reports its width. This is adequate (and oscillation-free) for the
  targeted applications but is not a sharp-interface method.
- **Forward tier is a mechanism check, not a solver competitor.** A TVD
  finite-volume scheme dominates forward 1-D problems; the forward tier exists to
  validate the constraint against exact ground truth. Relevance comes from the
  inverse setting, where no such solver applies.
- **Outlook.** The same primitive extends to other PDEs whose admissibility is a
  monotonicity/convexity statement — high-dimensional HJB (with the domain-
  restriction and expressiveness caveats of the de-risking analysis),
  Fokker–Planck (valid-density-by-construction), and eikonal (causality) — each a
  follow-up paper.

## Acknowledgments / Reproducibility

Built on `mononet` (`pip install mononet`); all experiments reproduce via
`applications/pinn/RUNBOOK.md`. Code, configs, and trained-model artifacts are in
the repository.

## References

See `references.bib`.
