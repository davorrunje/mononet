# Follow-up: `MonoResidual` gate collapse to a near-linear map (training pathology)

Status: Idea / bug-investigation backlog — not scheduled.
Origin: surfaced while building the Structure-Preserving PINN application
(`applications/pinn`, see `applications/pinn/FINDINGS.md`).

## What we observed

Fitting the 1-D Heaviside step with `absolute` mode, width 64, 10k Adam steps:

| construction | relu | softplus |
|---|---|---|
| plain `MonoLinear` × 4 | MSE 3.5e-4, slope 96 | MSE 3.0e-3, slope 17 |
| `MonoResidual` × 2 + head | **MSE 6.25e-2, slope 0.75** | **MSE 6.25e-2, slope 0.75** |

The `MonoResidual` stack **stays at a near-linear map** and never fits the step;
`relu` and `softplus` give **byte-identical** loss/slope, i.e. the activation is
inert — the block is sitting on its skip / near-linear path.

## Why it is a *training* problem, not an expressivity one

`MonoResidual` has **trainable gates** (`alpha_gate` skip, `beta_gate` dense) and
can recover the plain `MonoLinear` path exactly, so it is at least as expressive
as a `MonoLinear` stack. The failure is therefore **optimisation /
initialisation**: the gates initialise in (and remain stuck in) the skip/linear
regime, and gradients do not drive the dense path active. This matches the
`relu`≡`softplus` signature.

Note this is regime-dependent: `MonoResidual` performs fine on the tabular
benchmarks (loan size-ladder, large-dataset screen). The pathology showed up in a
narrow (width-1 input) sharp-feature regression. So it is an interaction between
gate init and this regime, not a blanket defect.

## What to investigate

- `MonoResidual` gate initialisation (`alpha_gate="shifted_elu"`,
  `beta_gate="scaled_elu"`): what is the effective skip/dense mix at init, and is
  the dense path's initial contribution ~0 (starving its gradient)?
- Whether a warm-start / gate-bias that makes the dense path active at init fixes
  it, without hurting the tabular results.
- Reproduce minimally (the Heaviside fit above) and add as a regression test.

## Scope

- mononet-core investigation (`mononet/*/layers.py`, `mononet/core/init.py`),
  **not** the PINN application. The PINN paper unblocks by using a plain
  `MonoLinear` stack (verified to fit sharply with softplus); this follow-up is
  about making `MonoResidual` train reliably in the sharp-feature regime.
- Consider filing as a GitHub issue when scheduled.
