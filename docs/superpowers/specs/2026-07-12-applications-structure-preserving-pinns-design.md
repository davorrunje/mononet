# Applications program — Structure-Preserving PINNs (Paper 1 + program map)

**Date:** 2026-07-12
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Depends on:** [Sub-project A](2026-06-27-A-core-algorithm-and-backends-design.md) (locked public layer API: `MonoLinear`, `MonoResidual`, `MonoInput`, `MonotonicityMask`, `MonoConfig`)
**Primary references:**
- Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>
- Sartor et al., *Advancing Constrained Monotonic Neural Networks*, ICML 2025 — <https://arxiv.org/abs/2505.02537>
- Raissi, Perdikaris & Karniadakis, *Physics-Informed Neural Networks*, JCP 2019
- LeVeque, *Finite Volume Methods for Hyperbolic Problems*, 2002 (entropy/TVD, Oleinik condition)
- Shi et al., *Physics-informed deep learning for traffic state estimation*, 2021 (soft-constrained PINN-TSE — the baseline to beat)

## 1. Context and the relevance problem

This spec introduces a new top-level area, `applications/`, for downstream
research papers built **on** `mononet` as a library. It is distinct from
`benchmarks/`, which reproduces the `mononet` paper's own tables. Nothing under
`applications/` ships in the PyPI wheel.

The seed idea proposed enforcing monotonicity **in the network architecture** so
that a PINN's solution is Total-Variation-Diminishing (TVD) and entropy-admissible
*by construction*, rather than via soft loss penalties that only approximate those
properties and turn training into an unstable multi-objective problem.

Two findings from brainstorming reshaped the scope:

1. **A forward 1-D problem alone is not relevant.** A TVD finite-volume solver
   dominates it on speed and accuracy. Relevance requires problems where a PINN is
   genuinely the right tool: high dimension (grids fail) or inverse/data-
   assimilation (a solver needs the full IC; a PINN does not).
2. **The monotonicity idea generalizes beyond hyperbolic PDEs.** The same
   architectural primitive enforces the *admissibility condition* of several PDE
   families, each of which is a monotonicity or convexity statement. This is a
   framework, not a single application.

A prior-art pass (deep-research, 2026-07-12; recovered from the run journal after
the auto-synthesis stage degraded — see §9) confirmed the gap and sharpened it:

- Conservation-law PINNs that target entropy/TV (WE-PINN) and shape-constrained
  PINNs (DC-PINN), and PINN traffic-state estimators (Shi et al. 2021), enforce
  these properties via **soft penalties**, not architecture. No published PINN
  embeds hard monotonicity in the *architecture* for these PDEs.
- Existing *hard* shape-constrained pricing networks (Dugas et al. 2001;
  Chataigner–Crépey "Deep Local Volatility" 2020; ICNN) are the **inexpressive
  convex-only special case** — `mononet`'s degenerate `s = (m, 0, 0)` branch
  (digest Corollary 3) — and pay a documented ~10× accuracy penalty. `mononet`'s
  convex/concave activation split is precisely the escape from that trade-off.
- HardNet (arXiv:2410.10807) and `mononet` establish that hard-constrained nets
  can be universal approximators, refuting the "hard constraints kill
  expressiveness" position that soft-penalty papers (e.g. Ackerer et al.) rely on.

## 2. Thesis and contribution

**Thesis.** A single architectural primitive — `mononet`'s hard, *expressive*
monotonicity/convexity — is a structure-preserving inductive bias that enforces
the admissibility condition of many PDEs **by construction**, across
forward/inverse and low/high dimension. Because each admissibility condition is a
monotonicity or convexity statement, one mechanism covers them all.

| PDE family | Admissibility condition | `mononet` constraint |
|---|---|---|
| Scalar conservation laws (Burgers, LWR, advection) | TVD + Oleinik entropy | monotone in `x` |
| HJB / stochastic control | comparison principle; admissible value function | monotone in state + convex |
| Fokker–Planck | valid density (non-negative, normalized) | monotone CDF |
| Eikonal | causality | monotone arrival time |

**Contribution of Paper 1 (framework + inverse-conservation-law flagship):**
1. The **admissibility abstraction** — a small interface mapping each PDE to
   (a) a `MonotonicityMask`/convexity spec, (b) an admissibility-violation metric,
   (c) an exact-or-reference ground truth.
2. A **problem registry** so each PDE is a plug-in module over a backend-agnostic
   core, with thin per-backend (JAX + PyTorch) trainers.
3. **Mechanism validation on the forward conservation-law tier** (exact ground
   truth): hard-monotone PINN vs. vanilla / soft-penalty / weight-clipping
   baselines and a TVD finite-volume + closed-form reference, cross-backend. The
   credibility anchor — the constraint yields TVD/entropy to machine precision on
   an exactly-solvable case *before* we deploy it on the harder inverse problem.
4. **The deep flagship — inverse scalar conservation laws / traffic state
   estimation.** Reconstruct a monotone-front density field `ρ(x, t)` (a queue /
   shock forming behind a bottleneck, incident, or signal — LWR traffic) from
   **sparse, noisy observations** (scattered probe-vehicle / loop-detector data),
   with hard monotonicity guaranteeing a **TVD, entropy-admissible, oscillation-
   free reconstruction by construction**. This is the headline: an *inverse*
   setting where PINNs genuinely beat classical solvers (a solver needs the full
   initial + boundary data; the PINN assimilates scattered points), attacking the
   most-cited named PINN failure — spurious oscillation at shocks — see §2a.

### 2a. Why this shape, and why it is impactful

Citation reality (Semantic Scholar, 2026-07-12): the PINN field is enormous
(Raissi et al. 2019 ≈ 18,166 cites) and therefore crowded — incremental soft-loss
variants vanish. What gets *noticed* is a **provable guarantee against a named
failure mode**, a result **where the PINN genuinely beats the classical
alternative**, and **reusable released code** (`mononet` is `pip install`-able).

The inverse-conservation flagship hits all three. The named failure is concrete:
PINNs (and PINN-based traffic state estimators, e.g. Shi et al. 2021) produce
spurious Gibbs oscillations near shocks/queues; a monotone-by-construction field
*cannot* oscillate. The PINN-beats-solver case is genuine: classical finite-volume
schemes cannot run without the full initial/boundary conditions, whereas the
inverse problem provides only sparse scattered observations — exactly the data-
assimilation regime PINNs own. And unlike the HJB alternative we de-risked (see
§9a), the monotonicity guarantee here is **unconditional within a cleanly-stated
problem class** (the monotone-solution class: single-front / queue / Riemann
scenarios), not a domain-restricted half-space caveat.

Hence: **one deep flagship, not a spread of shallow vignettes.** Breadth (HJB,
Fokker–Planck, eikonal) is delivered by the follow-up papers (§3), each at proper
depth.

## 3. Program map (paper series)

This spec fully designs **Paper 1** and records the map for the follow-ups. Each
follow-up gets its own brainstorm → spec → plan, reusing the framework.

- **Paper 1 — Framework + inverse-conservation flagship** (this spec): primitive +
  admissibility abstraction + registry + forward conservation-law mechanism tier
  (full) + deep inverse traffic-state-estimation result. Audience: scientific ML /
  numerical PDE / intelligent transport systems.
- **Paper 2 — High-dim HJB / stochastic control** (follow-up): admissible value
  functions by construction. **Carries the de-risking caveats (§9a):** monotonicity
  is domain-restricted (positive-orthant / non-negative state), the convex-HJB-by-
  construction lane is partly occupied (Liu et al. 2023), so the defensible
  contribution is the *conjunction* — joint monotone+convex, expressive beyond
  ICNN, high-dimensional, on the HJB residual, with a correct derived policy.
  Benchmark on naturally non-negative-state control (energy storage, inventory,
  epidemic) plus LQ (convex) and Merton (concave, 1-D validation).
- **Paper 3 — Fokker–Planck**: valid-density-by-construction; eliminates the
  negative-density failure of FP-PINNs; bridges to Sub-project D (flows).
- **Paper 4 — Eikonal / seismic traveltime**: causal-by-construction; geophysics
  inverse problems; beats established eikonal-PINN baselines.
- **Paper 5 — Expressive arbitrage-free surfaces** (separate, **not a PINN**):
  shape-constrained *regression* of price/IV surfaces. Different method and
  audience (quant finance). Sketched in §8; own brainstorm later.

## 4. Scope of Paper 1

### Goals
- Establish `applications/` (README, `_common`, conventions) as the home for
  research papers built on `mononet`.
- Ship the framework: admissibility abstraction, problem registry, backend-
  agnostic `core/`, per-backend trainers (JAX Flax NNX + PyTorch), Optuna search.
- Fully treat the **forward conservation-law mechanism tier** (exact ground
  truth): Burgers-Riemann, Burgers smooth→shock, linear advection, LWR traffic.
- Deliver the **deep inverse flagship — traffic state estimation**: reconstruct a
  monotone-front density field from sparse, noisy observations with a TVD/entropy-
  admissible, oscillation-free result by construction, benchmarked against a
  high-resolution TVD reference and against vanilla / soft-penalty PINN-TSE
  baselines under varying observation sparsity and noise.
- Executed notebook rendered into the Sphinx "Applications" nav.

### Non-goals
- No training loops, dataset loaders, or PINN code in the `mononet` wheel.
  Everything here is application-local.
- **Monotone-solution class only.** Both tiers restrict to problems whose entropy
  solution is monotone in `x` for all `t` (monotone initial data → shock; Riemann
  problems; single-front queue formation). Stated up front as the problem class.
  Explicitly out of scope and documented: non-monotone data (`u₀ = −sin πx`),
  rarefactions, N-waves, multi-front congestion. A continuous monotone network
  represents a shock as a steep-but-continuous ramp (zero overshoot), not a true
  discontinuity. (This class restriction is *unconditional within the class* —
  contrast the domain-restricted HJB monotonicity, §9a.)
- **No shallow vignettes.** HJB (Paper 2), Fokker–Planck (Paper 3) and eikonal
  (Paper 4) are *not* touched in Paper 1 beyond the registry being designed to
  admit them. Impact comes from depth on the inverse flagship, not breadth across
  thin demos (§2a).
- The forward conservation-law tier is a **mechanism check**, not a relevance
  claim: a finite-volume solver dominates forward 1-D scalar problems. Its role is
  exact-ground-truth validation of the constraint, nothing more. Relevance comes
  from the inverse flagship, where no solver competes.
- No adaptive/residual-based resampling (RAR) in the headline; optional, clearly
  flagged ablation only (it would help every method and confound the architecture
  comparison).

## 5. Architecture

### 5.1 Folder layout

```
applications/
├── README.md                     # what applications are; index; contrast with benchmarks/
├── _common/                      # shared, minimal until a 2nd app needs it
│   ├── __init__.py
│   ├── seeding.py                # deterministic RNG helpers
│   ├── metrics_io.py             # JSON results read/write (reuse benchmarks conventions)
│   └── plot_theme.py             # shared matplotlib theme
├── pinn/                         # Paper 1: framework + forward tier + inverse flagship
│   ├── README.md                 # abstract + headline result
│   ├── RUNBOOK.md                # exact commands to regenerate every figure/table
│   ├── paper/                    # Markdown paper scaffold (written before implementation, §11)
│   │   ├── paper.md              # full manuscript; results tables/figures filled in post-implementation
│   │   ├── figures/             # generated figures (committed; regenerable via RUNBOOK)
│   │   └── references.bib        # citations gathered during brainstorming/research
│   ├── __init__.py
│   ├── core/                     # backend-agnostic science (pure NumPy) — single source of truth
│   │   ├── admissibility.py      # AdmissibilitySpec: mask, convexity/concavity flags, violation metric
│   │   ├── problems/             # problem registry (one plug-in module per PDE)
│   │   │   ├── __init__.py       # registry: name -> Problem
│   │   │   ├── base.py           # Problem protocol (residual, admissibility, IC/BC or observations, ground truth)
│   │   │   └── conservation.py   # Burgers, linear advection, LWR — forward tier + inverse mode (Paper 1, full)
│   │   │   # hjb.py / fokker_planck.py / eikonal.py added by Papers 2 / 3 / 4 (registry ready; not in Paper 1)
│   │   ├── exact.py              # closed-form entropy solutions (Riemann; characteristics + R-H)
│   │   ├── reference_solver.py   # TVD finite-volume (Godunov) — forward ground truth + inverse reference field
│   │   ├── sampling.py           # deterministic collocation / IC / BC / eval sets; sparse-noisy observation sampler
│   │   ├── metrics.py            # L1/L2, admissibility violation, TV(t), overshoot, shock speed/position, mass
│   │   └── plotting.py           # profiles, TV(t) curves, error heatmaps, observation overlays
│   ├── models/
│   │   ├── protocol.py           # PINNModel protocol: build(problem, cfg) -> callable u(x, t)
│   │   ├── jax/                  # mononet.jax hard-monotone + vanilla + soft + weight-clip builders
│   │   └── torch/                # mononet.torch equivalents
│   ├── training/
│   │   ├── losses.py             # residual / IC / BC / data-fit term specs (backend-agnostic)
│   │   ├── jax_trainer.py        # jax.grad/hessian residual; optax; jit
│   │   └── torch_trainer.py      # autograd.grad(create_graph=True); torch optim
│   ├── configs/                  # one JSON per (problem × method × backend × seed × obs-sparsity/noise)
│   ├── experiments/
│   │   ├── run.py                # CLI: one (problem, method, backend, seed) -> results artifact
│   │   ├── search.py            # Optuna HP search (Typer CLI; mirrors benchmarks/_common/search.py)
│   │   └── sweep.py              # full matrix, using per-method tuned configs
│   ├── results/                  # committed metrics JSON + small figures; <problem>/reference.npz
│   ├── notebooks/
│   │   └── structure-preserving-pinn.ipynb
│   └── studies/                  # follow-up papers extend here (hjb/, fokker_planck/, eikonal/)
└── arbitrage/                    # Paper 5 (separate; regression, not PINN) — sketch only
```

Key invariants:
- `core/` is pure NumPy and imports **no** framework. It is the single source of
  truth that makes JAX and Torch numbers provably comparable.
- Framework code lives **only** in `models/{jax,torch}` and
  `training/{jax,torch}_trainer.py`. Baselines are alternate model builders behind
  one protocol.
- Adding a PDE = adding one registered `problems/*.py` module; no infrastructure
  change. This is what makes the follow-up papers incremental.

### 5.2 The admissibility abstraction

`core/admissibility.py` defines, per problem:
- `mask: MonotonicityMask` — sign per input axis (e.g. `x → −1`, `t → 0`).
- `convex_axes` / `concave_axes` — which inputs the solution is convex/concave in
  (drives `mononet`'s convex/concave activation split).
- `violation(u_grid) -> float` — a non-negative admissibility-violation measure
  (e.g. total positive part of `∂u/∂x` for a non-increasing target). The headline
  claim is that this is **0 by construction** for the hard-monotone model and
  **> 0** for soft/vanilla baselines.

### 5.3 Model construction

The solution field `u_θ(x, t)` is built from `mononet` layers with mask `x → −1`
(non-increasing; or `+1` per scenario), `t → 0`. Both tiers use the same field;
they differ only in the loss (data term present in the inverse tier).

**Architecture: `MonoResidual` blocks, shallow (≈4 layers total).** Use
`MonoResidual` blocks rather than a plain stack of `MonoLinear` layers, kept
deliberately shallow — on the order of 4 layers total. This follows the repo's own
depth-vs-scale finding that additional depth does not improve accuracy for this
construction (see `2026-07-05-deep-residual-accuracy-design.md`,
`2026-07-03-deep-monotonic-residual-design.md`, and the loan size-ladder
experiment); width/scale, not depth, is the lever. A shallow residual model also
keeps higher-order input derivatives (needed for the PDE residual) cheap and
well-conditioned. Depth is a config knob so the "depth doesn't help here either"
check can be reproduced as a small ablation, not re-litigated.

Cross-backend: JAX (Flax NNX) and PyTorch, using the locked Sub-project A API.
Baselines (same across tiers): **vanilla** (unconstrained MLP), **soft**
(unconstrained + TV/entropy penalty — the strawman), **weight-clip** (non-negative-
weight + monotone-activation, the inexpressive hard baseline), **hard-monotone**
(`mononet`, proposed). The inverse tier is additionally positioned against
published PINN-TSE (traffic state estimation) practice, which is unconstrained/soft.

### 5.4 Data generation

PINN "data" is generated **deterministically from a seed** at run time (no large
committed files):
- **Forward tier:** interior collocation, IC points `(x, 0)`, BC points, dense
  evaluation grid.
- **Inverse flagship:** interior collocation (for the residual) + a **sparse,
  noisy observation set** — scattered `(x_k, t_k, ρ_k + ε)` points emulating
  probe-vehicle / loop-detector data — and a dense evaluation grid for scoring.
  **No** full IC/BC is given; that is the inverse setting. A sweep over observation
  sparsity and noise level is the core experiment.

- Points generated once in NumPy (`core/sampling.py`); each backend converts the
  **same** arrays to its tensor type — JAX and Torch train on identical sets.
- Sampling and observation masks are **identical across all methods**, recorded in
  config. The architecture, not the data, must do the work.
- Ground truth: closed-form entropy solution where available; otherwise a
  high-resolution cached TVD finite-volume reference field
  (`results/<problem>/reference.npz`, small, committed, regenerable). Observations
  are subsampled from this reference field + seeded noise. No `datasets/` directory.

### 5.5 Constraint enforcement

- **Primary:** soft loss terms (residual + data-fit for the inverse tier; residual
  + IC + BC for the forward tier), **no** TV/entropy penalty — the architecture
  supplies admissibility. Clean apples-to-apples: every method shares the loss and
  differs only in architecture.
- **Ablation:** hard IC via an output ansatz `u = g + φ·N` where a monotonicity-
  preserving construction exists (forward tier), for the stronger "no soft
  constraints" story.

## 6. Metrics

Forward tier (vs. exact/reference):
- L¹ / L² error; **admissibility violation** (§5.2, headline: 0 for hard-monotone,
  > 0 for baselines); TV(t) evolution and near-shock overshoot (Gibbs); shock
  speed/position error; mass-conservation error.

Inverse flagship (vs. reference field, over the sparsity × noise sweep):
- **Reconstruction L¹ / L² error** as a function of observation sparsity and noise
  — the headline curves (where unconstrained/soft PINNs degrade and oscillate near
  the front, and the hard-monotone model holds);
- **admissibility violation** (near-shock oscillation / positive-`∂ρ/∂x` mass) → 0
  by construction vs. > 0 for baselines;
- shock/queue front position error; robustness across seeds and observation masks.

Cross-backend equivalence: JAX and Torch hard-monotone runs on identical points
agree within a documented tolerance — a secondary result dogfooding `mononet`'s
cross-backend guarantee.

### 6a. Hyperparameter search and fair-comparison budget

HP search uses **Optuna** (already a repo dependency, v4.9.0), reusing the
Phase-2a pattern (`benchmarks/_common/search.py` + a Typer CLI, `n_trials` budget,
seeded samplers). **Every method — vanilla, soft, weight-clip, hard-monotone —
gets the identical search space where applicable and the identical trial budget.**
This is load-bearing for the headline claim: "hard beats soft" is only credible if
the soft baseline was tuned at least as hard as the proposed model (the soft
penalty weight in particular must be searched, not fixed). Tuned per-method configs
are frozen to JSON and consumed by `sweep.py`; the search itself is a `slow`-marked,
RUNBOOK-documented step, not part of CI.

## 7. Testing, docs, dependencies

- **Testing.** `core/` unit-tested against closed forms (exact entropy solution; FV
  solver on a Riemann problem with known speed) to fixed tolerance; `sampling.py`
  determinism tests (same seed → identical arrays); a fast smoke-train per backend
  in CI under `-m "not slow"`; full training runs marked `slow`. Backends selected
  as elsewhere in the repo (`importorskip`).
- **Docs.** The notebook executes via myst-nb into a new Sphinx "Applications" nav
  section. Sphinx + myst-nb is the source of truth (not MkDocs).
- **Dependencies.** optax, torch optim, and the FV solver are **application-local
  dev dependencies**, never added to the `mononet` wheel. Optuna (v4.9.0) is already
  a repo dev dependency — reuse it, do not add another HP-search library. Preserve
  lazy backend imports.
- **Compute / devcontainer.** Develop on `default` (CPU, all backends — needed for
  both-backend code + the equivalence test). Run the heavy Optuna search and sweeps
  on **`gpu-jax`** (primary GPU backend; JAX `jit` for the repeated residual/Hessian
  — CPU is too slow for the search). The cross-backend result is the equivalence
  test plus reproducing the JAX-tuned config on Torch, not a second full search. No
  combined torch+jax GPU venv (CUDA-wheel conflicts).

## 8. Paper 5 sketch — arbitrage-free surfaces (separate, not a PINN)

Recorded for completeness; own brainstorm later. Shape-constrained **regression**
(not PDE-residual): expressive exact monotone+convex fitting of price/IV surfaces
vs. soft-penalty prior art (Ackerer et al.) and inexpressive hard prior art
(Dugas/Chataigner–Crépey, Bernstein-QP). Model-free core constraint: call price
`C(K)` decreasing and convex in **strike** `K`, monotone in maturity `T`. Spot-space
`delta ∈ [0,1]` / `gamma ≥ 0` is a **model-dependent** variant (Black–Scholes / 1-D
diffusion), to be flagged as such — strike-space monotonicity/convexity is the
model-free theorem, spot-space Greeks are not (see §9 rigor correction). Metric of
record: arbitrage-violation count → 0 by construction vs. residual violations for
soft methods.

## 9. Research provenance and rigor corrections

The deep-research pass (2026-07-12) verified claims adversarially (22 confirmed, 3
killed). Corrections folded into this spec:

1. **Do not equate prior hard nets with `mononet`.** The Dugas-2001 / ICNN /
   Chataigner–Crépey construction is the convex-only inexpressive special case, not
   `mononet`'s expressive activation split. Cite them as prior lineage; the novelty
   is expressiveness.
2. **Strike-space vs spot-space (Paper 5).** `C(K)` monotone-decreasing and convex
   in strike is a model-free no-arbitrage theorem. `delta ∈ [0,1]`, `gamma ≥ 0`,
   `vega ≥ 0` in spot are model-dependent (hold under Black–Scholes / 1-D diffusion;
   can fail under jumps / correlated stochastic vol). Keep the two separate.
3. **Expressiveness is defensible.** HardNet + `mononet` universal-approximation
   results refute "hard constraints kill expressiveness"; that refutation is the
   framing lever, not a claim to hedge.

The workflow's auto-synthesis output was placeholder-degraded; findings were
recovered from `journal.jsonl` (19 source extractions + 75 verification votes) and
manually verified. Several supporting citations were recent (2026) arXiv IDs not
independently re-fetched; the load-bearing prior art (Dugas 2001, Ackerer,
Chataigner–Crépey 2020, Cohen–Reisinger–Wang, ICNN, HardNet) is well established.

### 9a. HJB de-risking findings (2026-07-12) — recorded for Paper 2

Two focused research passes on the HJB option (before it was demoted to a follow-up)
surfaced findings that **Paper 2 must inherit**:

1. **Monotonicity is domain-restricted for HJB.** Convexity and monotonicity are
   independent properties. The clean high-D HJB value functions are symmetric bowls
   (LQ, Black–Scholes–Barenblatt `∝ ‖x‖²`): globally convex but monotone only on a
   half-space / positive orthant. A nondegenerate convex quadratic is never globally
   monotone. Merton is genuinely monotone + concave but effectively 1-D in state.
   The popular Han–Jentzen–E d=100 "LQG" benchmark (`g = ln((1+‖x‖²)/2)`) is
   **neither convex nor concave nor monotone** — a trap. ⇒ Paper 2 must use naturally
   non-negative-state problems (energy storage, inventory, epidemic; or positive-
   orthant pricing) and state the domain explicitly.
2. **The convex-HJB-by-construction lane is partly occupied.** Liu et al. 2023
   (arXiv:2309.09953) already use an ICNN-style convex net + a viscosity-solution
   theorem (convex-only, 1-D). ICNN value/Q-functions (Amos 2017), convex control
   (Chen–Shi–Zhang 2019), ICNN Lyapunov certificates (Manek–Kolter 2019), ISNN
   (joint monotone+convex, 2025, mechanics-only), and Bokanowski et al. 2026
   (structure-in-loss, not architecture) all border the idea. ⇒ Paper 2's defensible
   contribution is the **conjunction**: joint monotone+convex, *expressive beyond
   ICNN*, high-dimensional, on the HJB residual, with a correct derived policy — and
   it must explicitly distinguish those five works.

These are exactly why the flagship moved to inverse conservation laws, whose
monotonicity guarantee is unconditional within its problem class.

## 10. Open questions for the implementation plan

- **Inverse-flagship problem design.** LWR flux choice (Greenshields
  `Q(ρ) = v_max ρ(1 − ρ/ρ_max)`); which monotone-front scenarios (queue behind a
  bottleneck / red signal / incident); observation model (probe vs. loop-detector
  geometry); the sparsity × noise grid for the headline sweep.
- Whether a hard IC ansatz is meaningful in the inverse setting (likely forward-tier
  only, since the inverse problem has no full IC).
- LWR flux domain and BC handling for the forward tier.
- Tolerance thresholds for cross-backend equivalence on trained models.

## 11. Deliverable sequencing — paper scaffold first

Per the user's direction, work proceeds **manuscript-first**:

1. **Paper scaffold (Markdown, no results).** Write `pinn/paper/paper.md` as far as
   possible *before* implementation: title, abstract, introduction, related work
   (grounded in the §9/§9a research), the method (framework + admissibility
   abstraction + registry), the theory (TVD/entropy-by-construction statements and
   the monotone-solution-class scope), experimental design (benchmarks, baselines,
   metrics, sweeps), and results/discussion sections with **explicit placeholders**
   for numbers, tables, and figures. Gather `references.bib` from the research
   already done. This front-loads all reasoning that does not depend on runs and
   makes the implementation a matter of *filling in* pre-specified tables/figures.
2. **Implementation.** Build `core/` → models → trainers → experiments, TDD, per the
   plan produced by `writing-plans`.
3. **Fill results.** Execute the sweep, populate the placeholders, render the
   notebook into docs, finalize the manuscript.

LaTeX conversion is deferred (Markdown → LaTeX is mechanical). The plan from
`writing-plans` will make step 1 (scaffold) its first phase.
