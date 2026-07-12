# Structure-Preserving Physics-Informed Neural Networks: Hard Monotonicity as a Total-Variation-Diminishing Prior for Conservation Laws

**Authors:** Davor Runje (and collaborators — TBD)
**Status:** Draft scaffold — narrative complete; numerical results are placeholders (`[[…]]`) pending the experimental runs (see `RUNBOOK.md`).

> **Scaffold note (remove before submission).** This manuscript is written
> *before* the experiments per the project's manuscript-first workflow. Every
> claim about *what will be shown* is committed here so the implementation only
> has to fill in numbers. Placeholders are written as `[[NAME]]` and each has a
> matching planned experiment. If a result contradicts a claim below, the claim
> is revised to match the evidence — not the other way around.

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
and TVD finite-volume ground truth, where the admissibility violation is
`[[FWD-VIOLATION-HARD]]` for our method versus `[[FWD-VIOLATION-SOFT]]` for a
tuned soft-penalty baseline. We then apply it where a PINN genuinely beats a
classical solver — the *inverse* problem of traffic state estimation from sparse,
noisy observations — and show reconstruction error `[[INV-ERROR-HARD]]` versus
`[[INV-ERROR-SOFT]]` at `[[OBS-SPARSITY]]` observation coverage, with the
guarantee holding across backends (JAX and PyTorch agree to `[[XBACKEND-TOL]]`).
Code and the trained models are released as part of the `mononet` package.

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

> All values below are placeholders pending the runs; each has a matching planned
> experiment in `RUNBOOK.md`.

### 6.1 Forward mechanism tier

`[[TABLE-forward-tier]]` — per problem × method: L¹/L² error, admissibility
violation, overshoot, shock-speed error. *Expected:* hard-monotone violation and
overshoot ≈ 0; vanilla/soft show Gibbs overshoot `[[OVERSHOOT-SOFT]]`.

`[[FIG-tv-curve]]` — TV(t) for each method on Burgers-Riemann. *Expected:*
flat/non-increasing for hard-monotone; bumps for baselines.

`[[FIG-profiles]]` — solution profiles at `t = [[T-SNAP]]` near the shock.
*Expected:* clean monotone ramp (ours) vs ringing (baselines).

### 6.2 Inverse flagship — traffic state estimation

`[[FIG-inverse-sweep]]` — reconstruction L² vs observation sparsity and noise, all
methods. *Expected:* hard-monotone degrades gracefully; soft/vanilla degrade and
oscillate as data thins.

`[[TABLE-inverse]]` — reconstruction error, admissibility violation, front-position
error at representative (sparsity, noise) operating points.

`[[FIG-inverse-field]]` — reconstructed `ρ(x,t)` vs reference field with
observations overlaid, at `[[OBS-SPARSITY]]`.

### 6.3 Cross-backend equivalence

`[[TABLE-xbackend]]` — JAX vs PyTorch hard-monotone agreement (max abs difference
`[[XBACKEND-TOL]]`) on identical points.

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
