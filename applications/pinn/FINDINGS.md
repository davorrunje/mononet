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

### DECISIVE TEST — direct supervised fit (no PDE). Architecture, confirmed.

Fit the `hard_monotone` field *directly* to the exact solution (supervised MSE on
a 128×64 grid, width 64 × 3 blocks, 6000 steps, Adam 2e-3):

```
final supervised MSE: 0.0637   DIRECT-FIT L1=6.774e1  L2=4.534e0
```

Direct fit is **no better than the PINN** (L1 67 vs 52). So the field **cannot
represent the solution even with the answer in hand** ⇒ **expressivity /
architecture problem, not PINN training and not optimization.**

**Root cause (architecture).** The field is
`MonoStack(concat[x, h(t)])` where `MonoStack` is monotone in `x` **and** monotone
in each channel of a free `t`-embedding `h(t)` (mask `[sign_x, +1,…,+1]`; see
`models/jax/builders.py:HardMonoField`). The Burgers-Riemann solution is a **shock
front that translates in `x`** as `t` grows (`x_s(t)=s·t`). Making a sharp
monotone front *move horizontally* requires `x` and `t` to combine so the
transition location shifts with `t`; the current weak concat→monotone-stack
coupling saturates at a poor fit and does not improve with capacity/steps. The
constraint is satisfied (viol≈0) but the front can't move → large L1/L2.

### Refined next steps
1. **Confirm the front-translation cause (cheap, decisive).** Run
   `burgers_riemann` with `uL=-uR` (shock speed `s=0`, **stationary** front). If
   `hard_monotone` fits the stationary shock well but fails moving ones, the
   diagnosis is nailed to front-translation coupling.
2. **Fix the `x`–`t` coupling** so a monotone-in-`x` front can translate:
   - inject a learnable **monotone shift** of `x` by `t` inside the stack
     (e.g. feed `sign_x*x + g(t)` where `g` is an unconstrained MLP of `t`), so
     the transition location moves with `t`; or
   - condition the monotone-in-`x` stack on `t` more richly (t-modulated
     weights/bias) rather than a low-dim concatenated embedding.
3. **Re-evaluate the inductive bias.** "Monotone in `x`, free in `t`" is correct
   *pointwise per `t`*, but the moving front is the hard part — confirm the fixed
   field can represent it before running the full sweep or the inverse flagship.
4. Sanity: confirm grid/normalisation so absolute L1 magnitudes are interpretable.

**Bottom line:** MonoResidual is not "bad at monotone problems" — the *field
wiring* can't move a shock. This is a fixable architecture issue, and it should be
fixed (and re-validated by direct fit) **before** any HP search or the inverse
flagship.
