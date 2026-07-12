# PINN Paper — Findings & Next Steps (working notes)

Durable log of empirical findings so they survive across sessions/clones. Not
part of the manuscript; feeds §6/§7 once resolved. Newest first.

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
