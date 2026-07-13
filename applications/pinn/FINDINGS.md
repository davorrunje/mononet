# PINN Paper — Findings & Next Steps (working notes)

Durable log of empirical findings so they survive across sessions/clones. Not
part of the manuscript; feeds §6/§7 once resolved. Newest first.

## 2026-07-13 — MonoResidual restored (PR #100 fixes the gate collapse)

PR #100 (near-zero gate init + softplus gate) fixes the `MonoResidual` collapse.
Smoke test (GPUs free):

- **1-D Heaviside:** MonoResidual now fits *sharper* than the plain MonoLinear
  fallback — relu MSE 3e-5 (slope 197), softplus MSE 4e-4 (slope 94) — vs the old
  collapse (MSE 0.0625, slope 0.75, relu≡softplus).
- **Inverse PINN (LWR, tuned-with-plain-HP, 4 seeds):** MonoResidual field L2 IQM
  **0.66** vs plain-MonoLinear **0.79**, violation 0 — ~17 % better *before*
  re-tuning.

`HardMonoField` switched back to `MonoResidual` (spec's ~4-layer design) in **both**
backends; also added the input normalization to the torch builder for parity (jax
already had it) — both backends now MonoResidual + normalized. Gate-collapse
follow-up marked resolved. **Re-running the headline + sweep with the MonoResidual
field next; the results below will be superseded.**

## 2026-07-13 — Sparsity x noise sweep: admissibility gap widens with noise; accuracy a wash

`results/inverse-sweep.json` (tuned-once, stress-tested; 4 obs-counts x 3 noise,
4 seeds IQM). Representative (n_obs=80, noise=0.10):

| method | L2 | violation |
|---|---|---|
| hard_monotone | 1.01 | **0.000** |
| vanilla | 0.94 | 0.365 |
| soft | 1.08 | 0.510 |
| weight_clip | 2.18 | 0.000 |

- **Monotonicity violation is the robust, noise-amplified discriminator:**
  `hard_monotone` is **0 everywhere**; vanilla/soft grow from ~0.02–0.06 (no noise)
  to **0.28–0.56** (noise 0.10). The admissibility gap *widens* exactly as data
  degrades — the "robust where data is bad" story, in the admissibility dimension.
- **L2 accuracy stays a wash** across the whole grid (hard vs vanilla within
  ~10 %); the constraint neither wins nor loses on error even at high
  noise/sparsity. Opportunity #1 (accuracy win under sparsity) did **not** hold.
- **`oob_frac` did not discriminate** (all 0.3–0.6): it counts *tiny* excursions
  (hard_monotone shows oob 0.61 with overshoot 0.002), so it is a poor
  physical-validity proxy as defined. **Follow-up:** measure excursions outside
  the *physical* bounds `[0, ρ_max]` (negative density / over-jam), not the field's
  own range — that should separate oscillating baselines from the monotone field.

Net: the sweep **strengthens the admissibility claim** (guaranteed zero violation
vs. baselines that oscillate more as noise grows) but confirms **accuracy parity,
not superiority**. Paper §6.2 updated with the violation-vs-noise panel.

## SESSION SUMMARY (2026-07-12, overnight autonomous run)

Started from "MonoResidual gives disappointing PINN results." Root-caused a stack
of *usage/training* issues (mononet itself is fine — Heaviside/UAP), fixed them,
and firmed up a paper-grade headline.

**Fixed (committed):** (1) `MonoResidual` field collapsed to ~linear → switched to
a plain `MonoLinear` stack; (2) missing input normalization → added (tied to
`absolute`-init scale); (3) inverse-tier residual divergence → global-norm
gradient clipping; (4) forward residual smears shocks (known PINN issue — forward
is a mechanism check only).

**Headline (10-seed IQM, equal-budget tuned):** on the inverse/traffic flagship,
expressive hard monotonicity is **accuracy-competitive** with unconstrained/soft
PINNs (L1 IQM 3.17 vs 3.9; bands overlap, L2 a wash) and the **only structurally
admissible, oscillation-free** solution (violation 0 vs 0.08–0.10). Inexpressive
`weight_clip` fails (L1 22.5) → it is the *expressiveness* that matters. Forward
tier: constraint costs ~20 % accuracy but gives zero oscillation (the worst case).

**Artifacts:** `results/{inverse-headline,forward-mechanism}.json`;
reproducible `experiments/headline.py` + RUNBOOK; `paper/paper.md` §6 filled +
abstract reworded to the honest claim. ruff/mypy clean; 68 tests pass.

**Open (need a different env / more work):** cross-backend equivalence (needs
Torch → `default` container); figures + sparsity×noise sweep; the two mononet-core
/ docs follow-ups (`MonoResidual` gate-collapse; input-normalization docs).

## 2026-07-12 — Forward mechanism panel (10-seed IQM) + paper §6 filled

Forward `burgers_riemann`, same protocol (`results/forward-mechanism.json`):

| method | L1 IQM [95%] | L2 | violation | overshoot |
|---|---|---|---|---|
| hard_monotone | 6.15 [4.41, 8.37] | 1.45 | **0** | **0.003** |
| vanilla | 5.12 [4.31, 6.31] | 1.06 | 0.232 | 0.127 |
| soft | 5.50 [5.12, 6.96] | 1.15 | 0.106 | 0.019 |
| weight_clip | 46.0 | 3.79 | 0 | 0.42 |

Forward is the constraint's **worst case**: hard_monotone costs ~20 % L1/L2 (the
strong residual smears the shock) but is the only oscillation-free (overshoot
0.003 vs vanilla 0.127), zero-violation solution. Contrast with the inverse tier
(below), where accuracy is a wash and admissibility is free — the paper's arc.

`paper/paper.md` §6.1/§6.2 tables filled from the two JSON artifacts; abstract
reworded to the honest claim (competitive accuracy + sole admissibility, not an
accuracy win). Pending in the manuscript: figures, sparsity×noise sweep,
cross-backend check (needs `default` container for Torch).

---

## 2026-07-12 — FIRMED headline (10-seed IQM): accuracy a wash, admissibility the win

Repo-protocol aggregation: tune once (20 trials), evaluate **10 seeds**, report
**IQM** (trim `n//4`) with a **95 % bootstrap band**. Inverse `lwr_riemann`,
8000 steps, grad-clip. `results/inverse-headline.json`.

| method | L1 IQM [95%] | L2 IQM | violation | overshoot |
|---|---|---|---|---|
| **hard_monotone** | **3.17 [2.37, 3.80]** | 0.79 | **0** | 0.023 |
| vanilla | 3.92 [3.34, 4.27] | 0.73 | 0.083 | 0.024 |
| soft | 3.95 [3.36, 4.37] | 0.72 | 0.104 | 0.040 |
| weight_clip | 22.5 [22.0, 23.1] | 2.10 | 0 | 0.30 |

**Honest correction to the single-seed run** (which showed L1 1.48 — optimistic).
Over 10 seeds: `hard_monotone` L1 is lowest but its band **overlaps** the
baselines, and its L2 is **marginally worse**. So **accuracy is a wash** with
unconstrained/soft. The **decisive, non-overlapping result is admissibility**:
violation **0** vs 0.08–0.10 (baselines oscillate), and `weight_clip` (L1 22.5)
confirms it is the *expressive* constraint that matters, not any monotone one.

**Defensible claim:** *on the inverse/traffic problem, expressive hard
monotonicity gives reconstruction accuracy competitive with unconstrained/soft
PINNs while being the only structurally admissible, oscillation-free solution
(zero monotonicity violation by construction).* Do NOT claim it beats on accuracy.

---

## 2026-07-12 — HEADLINE: equal-budget inverse search — mononet wins the flagship (single-seed; superseded)

Equal-budget Optuna (12 trials/method, tier-aware space incl. `residual_weight`/
`data_weight`, grad-clip 1.0), `lwr_riemann` inverse (reconstruct traffic density
from 80 sparse noisy observations), 6000 steps:

| method | L1 | L2 | violation | overshoot | tuned (lr, w, res, data) |
|---|---|---|---|---|---|
| **hard_monotone (mononet)** | **1.48** | 0.55 | **0** | **0.014** | 1.7e-3, 32, 0.12, 7.5 |
| vanilla | 2.52 | 0.54 | 0.078 | 0.011 | 1.7e-4, 64, 0.37, 6.8 |
| soft | 2.48 | 0.44 | 0.111 | 0.054 | 2.5e-3, 64, 0.04, 4.3 |
| weight_clip (naive hard) | 24.0 | 2.13 | 0 | 0.29 | 3.6e-3, 32, 0.24, 36 |

**Result:** on the inverse/traffic flagship, **mononet's expressive hard
monotonicity gives the best reconstruction L1 (1.48, ~40 % better than the ~2.5
baselines) AND zero admissibility violation with the least oscillation.** The two
foils land as the thesis predicts:
- *unconstrained / soft* — decent error but **violate monotonicity** (0.08–0.11)
  and oscillate; soft's penalty only softens it.
- *weight_clip (inexpressive hard)* — admissible but **cannot fit** (L1 24). So it
  is **not** "any monotone constraint" that works — it is the *expressive* one.

The search also tuned **`residual_weight` low** (0.12 mono / 0.04 soft), leaning on
the data term — consistent with the strong residual being counterproductive; the
architecture supplies the structure instead.

**Arc closed:** the forward tier is the constraint's worst case (residual smears
shocks; mechanism check only); the **inverse data-assimilation regime the paper
targets is where expressive hard monotonicity wins on both accuracy and
admissibility.** This is the paper's headline, and it required: plain `MonoLinear`
field, input normalization, gradient clipping, and equal-budget HP tuning — all
now in place.

---

## 2026-07-12 — BREAKTHROUGH: gradient clipping fixes the inverse divergence

The inverse divergence was a gradient-scaling issue. Added **global-norm gradient
clipping** to the trainers (`grad_clip`, optax `clip_by_global_norm`), extended
`run_one` + `search.py` to the **inverse tier** and added **`residual_weight`**
(and `data_weight`) to the search space.

Inverse `hard_monotone` on `lwr_riemann` (4000 steps, lr 5e-3, res 1, data 10):

| grad_clip | L1 | L2 | viol | overshoot |
|---|---|---|---|---|
| 0.0 | 34.6 | 2.87 | 0 | 0.53 |
| **1.0** | **4.55** | **0.61** | **0** | **0.05** |

With clipping, `hard_monotone` reconstructs the traffic field **competitively with
vanilla (L1 3.85) / soft (3.46)** — and is the **only oscillation-free, admissible
one** (viol 0, overshoot 0.05). This is the regime the thesis targets, and the
constraint now pays off. Next: the equal-budget **inverse** Optuna search across
all methods to confirm fairly.

---

## 2026-07-12 — Optuna HP search (forward tier): helps, gap remains

First actual HP search (all prior runs were hand-picked). Equal-budget Optuna,
`burgers_riemann` forward, 20 trials, 5000-step template:

| method | best lr | width | ic_wt | L2 | L1 | viol |
|---|---|---|---|---|---|---|
| vanilla | 9.1e-3 | 16 | 1.7 | 1.02 | 6.03 | 0.23 |
| soft | 5.1e-3 | 32 | 4.8 | 0.90 | 4.45 | 0.27 |
| hard_monotone | 9.9e-3 | 16 | 43 | 2.78 | 25.9 | 0 |

Tuning **helped** (`hard_monotone` L1 52→26) and did **not** diverge on the
forward tier. But `hard_monotone` stays ~4–6× less accurate than vanilla/soft
while being the only admissible model (viol 0). On the forward tier this is a
genuine structure-vs-accuracy trade (and the forward tier is only a mechanism
check).

**Caveats on this search:** (1) `search.py` is **forward-only** (`run_one` builds
IC/BC) — the **inverse/traffic flagship was not searched**; (2) the space is
`lr`/`width`/`ic_weight` only — it does **not** tune `residual_weight` or `steps`,
the knobs implicated in forward shock-smearing and the inverse divergence.
**Next:** extend the search to the inverse tier and to `residual_weight` (and add
stabilisation), then re-judge the flagship.

---

## 2026-07-12 — Inverse/traffic: residual DESTABILISES the constrained field

Ran the inverse flagship — reconstruct LWR density from 80 sparse noisy
observations (residual + data loss, no IC/BC), 8000 steps:

| method | L1 | L2 | viol | overshoot |
|---|---|---|---|---|
| vanilla | 3.85 | 0.66 | 0.16 | 0.03 |
| soft | 3.46 | 0.64 | 0.05 | 0.04 |
| weight_clip | 38.4 | 2.85 | 0 | 0.07 |
| **hard_monotone** | **124** | 9.3 | 0 | **2.34** |

`hard_monotone` did **not** improve on the inverse problem — it *blows up*
(predicts values far outside the density range). My "inverse favours the
constraint" hypothesis is **not confirmed.**

**Characterisation (hard_monotone, inverse):**

| config | L1 | pred [min,max] | final loss |
|---|---|---|---|
| data-only (res 0), lr 5e-3 | 32.9 | [-0.15, 1.19] | 0.29 (stable) |
| data+res, lr 1e-3 | 336 | [-6.5, 10.0] | **1892 (diverged)** |
| data-heavy (res 0.1, data 50), lr 1e-3 | 137 | [-3.5, 5.0] | 43.7 (diverging) |

(ref density ∈ [0.2, 0.8].) **The PDE residual term causes training divergence
for the constrained field** — with residual on, the loss *grows* and predictions
run away to ±10; data-only is stable but mediocre (L1 33). Vanilla trains fine
with the *same* residual+data setup, so this is specific to the mononet field:
a runaway feedback (u grows → `flux_prime(u)·u_x` grows → residual grows). Likely
an ill-conditioned residual-gradient / step-size interaction with the
`absolute`-mode field, **plausibly fixable** with standard stabilisation
(gradient clipping, lr schedule/lower lr, residual loss balancing / normalisation,
or a weak-form residual) — but currently an open blocker for the constrained PINN.

**State:** the constrained field can be *fit* (direct supervised: mean|err| ~0.1)
but cannot yet be *trained through the PDE residual* without diverging. This — not
mononet, architecture, normalization, or expressivity — is the real blocker for
both tiers. Next decision: which stabilisation to try.

---

## 2026-07-12 — Forward-tier gap diagnosed: strong-form residual smears the shock

Loss-weight sweep, `hard_monotone`, `burgers_riemann`, 8000 steps lr 5e-3
(normalized field):

| weights | L1 |
|---|---|
| ic-only (res 0, ic 10, bc 1) | **40.0** (best) |
| ic-heavy (res 1, ic 100, bc 10) | 43.6 |
| default (res 1, ic 10, bc 1) | 52.9 |
| balanced (res 1, ic 1, bc 1) | 63.1 |
| res-heavy (res 10, ic 1, bc 1) | 64.8 (worst) |

**The residual hurts** — more residual weight → worse. And direct-fit of the same
(fixed, normalized) field reaches mean|err| 0.084 (stationary) / 0.11 (moving),
so the **field can represent the shock**; moving vs stationary is a minor gap (not
front-translation). Therefore the forward-tier `hard_monotone` gap is the
**strong-form PDE residual smearing the near-discontinuity** — minimising
`u_t + u·u_x` at the shock rewards flattening it. This is a *known, fundamental*
PINN-for-conservation-laws problem (the classical PDE holds only away from the
shock; the entropy/weak form is needed — cf. WE-PINN), **not** a mononet, field,
or normalization issue.

**Important implication for the paper.** The forward tier was always just a
mechanism check; this confirms it is the *wrong* setting for the constrained field
(the residual fights the shock). The **inverse flagship (traffic)** should be far
more favourable: scattered interior *observations* anchor the solution and the
residual is a lighter regulariser — the regime the paper's thesis actually targets.
**Next: test hard_monotone on the inverse/traffic problem** (with data), where the
constraint's oscillation-free guarantee should pay off. Also consider a weak/
entropy residual for the forward tier (secondary).

---

## 2026-07-12 — Applied fixes + remaining gap (latest)

**Done:** (1) `HardMonoField` rebuilt from a plain `MonoLinear` stack (the
`MonoResidual` gate-collapse is a separate mononet-core follow-up); (2) **input
normalization to `[-1,1]` added to all JAX fields** (inside the model, so the PINN
residual's autodiff chains through it). mononet's `absolute` init assumes
~unit-scale inputs — recorded as a docs follow-up
(`2026-07-12-mononet-input-normalization-docs-idea.md`).

**Head-to-head (normalized, 10k, lr1e-2), stationary step, direct fit:** plain
`MonoLinear` mean|err| **0.0195** vs `MonoResidual` **0.2157** — confirms the
collapse is `MonoResidual`-specific, not normalization/training. Hypothesis "same
problem" **refuted**.

**PINN comparison, `burgers_riemann`, 8000 steps, lr 5e-3 (normalized):**

| method | L1 | L2 | viol | overshoot |
|---|---|---|---|---|
| vanilla | 3.96 | 0.94 | 0.20 | 0.12 |
| soft | 4.75 | 1.29 | 0.05 | 0.015 |
| weight_clip | 49.6 | 4.20 | 0 | 0.28 |
| **hard_monotone** | 52.9 | 4.20 | 0 | 0.21 |

Normalization + training **fixed the baselines** (vanilla/soft L1 ~10→~4) but
**`hard_monotone` is still stuck (~53)** — yet *direct supervised fit* of the same
normalized field reaches mean|err| 0.02. **So the remaining `hard_monotone` gap is
PINN training** (residual + soft IC/BC), NOT representation and NOT the field: the
constrained field *can* fit the solution, but the PINN loss/optimisation doesn't
drive it there. Next: diagnose the PINN-training gap for the constrained field
(loss weighting, IC/BC handling, moving-front residual) — the constraint likely
interacts badly with the current soft-IC/BC objective.

---

## 2026-07-12 — VERIFIED ROOT CAUSE + fix (supersedes everything below)

Established the correct baseline first (per author guidance): a **4-layer
`absolute`-mode mononet built from plain `MonoLinear` layers fits the 1-D
Heaviside** — ReLU MSE 3.5e-4 (slope ~96), and **softplus MSE 3e-3 (slope ~17)**,
a genuine sharp step. mononet is a UA here, as expected (Lean proof).

**Decisive controlled test (same Heaviside, width 64, 10k steps):**

| construction | relu | softplus |
|---|---|---|
| plain `MonoLinear` × 4 | MSE 3.5e-4, slope 96 | MSE 3.0e-3, slope 17 ✅ |
| `MonoResidual` × 2 + head (app-style) | MSE 6.25e-2, slope 0.75 | MSE 6.25e-2, slope 0.75 ❌ |

In training, the app's `MonoResidual`-based field **stays at a near-linear map**
(MSE 0.0625 — the *same* value the PINN app plateaued at; `relu` and `softplus`
byte-identical). A plain `MonoLinear` stack does not.

**Interpretation (corrected — per author).** `MonoResidual` has **trainable
gates** and can recover the plain `MonoLinear` path, so it is **at least as
expressive** as `MonoLinear`. The failure is therefore **optimization /
initialization**, not expressivity: the gates initialise in (and stay stuck in)
the skip/near-linear regime and training doesn't drive them into the dense path.
The `relu`≡`softplus` byte-identity is the signature — on the collapsed skip path
the activation is inert. So mononet, the activation, and PINN training are all
fine; the issue is the `MonoResidual` **gate optimisation** in this regime.

**Open fork (needs a decision):**
1. **Diagnose/fix the gate optimisation** — inspect `MonoResidual` gate init
   (`alpha_gate`/`beta_gate`) and why gradients don't move them here; make the
   dense path active at init. Preserves the residual architecture the spec chose.
2. **Pragmatic:** build `HardMonoField` from a plain `MonoLinear` stack (verified
   to fit sharply with softplus) and revisit residual blocks later.

Re-validate whichever path by direct fit to the Burgers solution, then re-run the
method comparison.

---

## 2026-07-12 — CORRECTION (supersedes conclusions below)

**mononet is NOT the limitation.** Per the author + the Lean UAP proof
(<https://davorrunje.github.io/neural-network-proofs/>), mononet approximates the
Heaviside/step function and is a universal approximator of monotone functions
with ~4 layers. Therefore:

- The "bare mononet can't sharpen a step" result below is an artefact of a
  **broken ad-hoc test harness** (the tell: softplus and elu gave *byte-identical*
  MSE/slope — the activation wasn't engaging), **not** a mononet property.
  **Retracted.**
- What remains genuinely observed (via the real `run.py`/`jax_trainer` pipeline):
  `hard_monotone` fits `burgers_riemann` far worse than `vanilla` and renders a
  near-linear ramp. Since mononet *can* represent the target, this is a
  **usage/setup bug in the PINN app or my measurement**, to be found — leading
  candidates: **unnormalised inputs** (raw `x∈[-2,3]`, `t∈[0,1.5]` fed straight to
  the `absolute`-mode stack; the repo's benchmarks standardise inputs), the
  `HardMonoField` wiring, or the loss/metric scaling.
- **Next step (disciplined):** reproduce monotone fitting through a *known-good*
  path (repo benchmark trainer / a documented example that fits a monotone target)
  and compare against the PINN app's usage — input scaling first. Do **not** draw
  architecture conclusions from bespoke harnesses again.

The section below is kept for the record but its mononet-implicating conclusions
are superseded by this correction.

---

## 2026-07-12 — Rediscovered: hard-monotone (MonoResidual) underfits the shock

**Setup.** `burgers_riemann` (forward tier, exact ground truth), JAX/GPU, seed 0,
soft IC/BC loss, default architecture unless noted. Via
`applications.pinn.experiments.run.run_one`.

**Result — the four methods (2000 steps, width 32 × 2 blocks):**

| method | L1 | L2 | admiss. violation | overshoot |
|---|---|---|---|---|
| vanilla | 9.13 | 1.50 | 0.128 | 0.029 |
| soft | 9.18 | 1.48 | 0.087 | 0.007 |
| weight_clip | 126.7 | 8.17 | **0** | **0** |
| **hard_monotone** | 52.6 | 4.02 | **0** | 0.298 |

The hard-constrained models are admissible (violation ≈ 0) but **~5× (MonoResidual)
to ~10× (weight-clip) worse in L1/L2** than the unconstrained/soft baselines. The
"hard beats soft" thesis fails here on accuracy.

**Diagnostic — is it under-training / under-capacity? NO.** Scaling
`hard_monotone` up leaves it flat:

| width | blocks | steps | L1 | L2 |
|---|---|---|---|---|
| 32 | 2 | 2 000 | 5.265e1 | 4.022e0 |
| 64 | 3 | 8 000 | 5.278e1 | 4.054e0 |
| 128 | 4 | 16 000 | 5.278e1 | 4.054e0 |

Identical to 3 sig-figs across a 4×-wider, 8×-deeper, 8×-longer sweep ⇒ it
converges to the **same degenerate solution** regardless of capacity/budget. This
is an **optimization/architecture pathology**, not a tuning gap.

**Caveats.** Single seed; no Optuna HP search yet; forward tier only (a mechanism
check, not the inverse flagship); default loss weights (`ic=10, res=1, bc=1`).

### Direct supervised fit (no PDE) — isolates architecture from PINN training

Fit the `hard_monotone` field *directly* to the exact solution (supervised MSE,
128×64 grid, width 64 × 3 blocks, 6000 steps, Adam 2e-3): L1 67, MSE 0.064 — **no
better than the PINN.** So it is **not** a PINN-training problem.

### ⚠️ Correction: the "front-translation" hypothesis was FALSIFIED

Direct-fit to a **stationary** shock (`u_l=0.5, u_r=-0.5`, s=0 — a *t-constant*
monotone step) fits **just as badly** as the moving shock:

```
stationary (s=0)   MSE=0.0673  L1=6.96e1   mean|err|=0.216
moving     (s=0.5) MSE=0.0637  L1=6.77e1   mean|err|=0.216
```

So the problem is **not** about moving the front. The field can't fit even a
*static* monotone step. (My earlier committed root-cause — front translation —
was wrong; the stationary test refuted it.)

### Actual finding: an HP-invariant collapse to a near-linear ramp

Profile at any `t` (stationary step, ref = ±0.5 hard step at `x0`):

```
x:    -2.0  -1.37  -0.74  -0.11 | 0.52  1.15  1.78  2.41
pred: +0.61 +0.43 +0.25 +0.07 |-0.11 -0.29 -0.47 -0.65   (near-LINEAR ramp)
ref:  +0.5  +0.5  +0.5  +0.5  |-0.5  -0.5  -0.5  -0.5    (sharp step)
```

The field is monotone-decreasing (constraint OK, bounded output) but renders the
step as an **almost-affine ramp across the whole domain** — it never sharpens.
And the fit is a **representational fixed point**: MSE=0.0673 / mean|err|=0.2157
are **identical to 4 dp across** lr ∈ {2e-3, 1e-2, 3e-2}, steps ∈ {6k, 20k},
width ∈ {64, 128}, blocks ∈ {3, 4}. Capacity and optimizer aggression change
*nothing* ⇒ not under-training, not optimization — the wired field simply cannot
(or will not) produce a steep monotone transition.

**Metric-scale note:** the reported `l1`≈9 (vanilla) … 69 (hard) are on an
inflated scale; the honest quantity is **mean|err|**: ~0.03 (vanilla) vs ~0.22
(hard) on a range-1 signal. Normalise `metrics.l1/l2` so numbers are
interpretable.

### Refined next steps (revised)
1. **Isolate mononet from the app wiring.** Fit a *minimal* `mononet` stack
   (`MonoInput` + `MonoLinear`/`MonoResidual`, 1-D `x` only, no `t`) to a sharp
   monotone step. Can *any* config produce a steep transition? Sweep
   `mode ∈ {absolute, switch}`, `mono_activation`, depth. → tells us whether the
   ceiling is in **mononet itself** or in `HardMonoField`'s wiring.
2. **Explain the HP-invariance.** Why byte-identical across capacity? Inspect
   `HardMonoField`/`MonoResidual` for a collapse (blocks → identity, linear head,
   activation saturation, init scale). Possible bug or a real construction limit.
3. **Consequence for the thesis.** If smooth monotone nets *fundamentally* can't
   render near-discontinuities sharply, that is itself a key paper finding and
   forces a rethink (sharper activation, coordinate/scale transform, or accepting
   a quantified ramp width) — resolve **before** HP search or the inverse flagship.
4. Fix the `metrics.l1/l2` normalisation.

**Bottom line (corrected):** the failure is **not** front-translation and **not**
training budget. The `hard_monotone` field converges — invariant to capacity and
LR — to a near-linear monotone ramp and cannot sharpen a step. Next diagnostic
must isolate whether this ceiling lives in `mononet` or in the app's field wiring.
