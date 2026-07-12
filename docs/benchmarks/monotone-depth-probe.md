# Synthetic monotone-depth probe

Does depth ever help constrained monotone networks? A synthetic investigation
of the depth-null empirical result from the [large-dataset
screen](large-dataset-screen.md) using monotone targets with a controllable
complexity knob.

## Motivation

Across the [loan size-ladder](loan-size-ladder.md) (no dose-response in N) and
the 5-dataset max-size [large-dataset screen](large-dataset-screen.md), deep
`absolute`-residual monotone stacks never beat shallow ones. This robust
empirical null leaves three possible causes:

1. **(E) Expressivity/prior.** Depth-separation theorems for unconstrained ReLU
   nets rely on **oscillatory folding** (Telgarsky 2016; Eldan & Shamir 2016;
   Montúfar et al. 2014) to multiply complexity with depth — a mechanism
   **monotonicity forbids**. The CMNN construction is a universal approximator
   of monotone functions at shallow depth (Runje & Shankaranarayana 2023; Sartor
   et al. 2025), so depth may add no *representable* function the monotone
   target needs.

2. **(D) Data simplicity.** Credit/income targets are near-additive (low-order
   interactions); even if depth helped complex monotone targets, real datasets
   give it nothing to exploit.

3. **(O) Optimization.** Deep constrained (non-negative-weight) stacks may be
   hard to train. Residual connections prevent degradation but create no depth
   *advantage*.

This probe **separates them** using synthetic *monotone* targets with a
controllable complexity knob `c ∈ {1, 2, 4, 8}`. Three target families:

- **Additive control** (`c` ignored): f = Σᵢ gᵢ(xᵢ), gᵢ random monotone
  piecewise-linear. Depth must not help — anchors the null.

- **Teacher-depth sweep** (the acid test): the target *is* a random **monotone
  teacher network** of depth `c`, built as a seeded monotone MLP
  (non-negative weights + softplus activation). The target is monotone by
  construction; a deep student of matching size is a universal monotone
  approximator and can approximately represent it — so persistent deep-student
  error approximately isolates **(O)** from **(E)**.

- **Max/min-lattice teacher** (most depth-favoring): f = nested max/min of
  monotone terms, `c` = nesting depth. Max/min composition builds piecewise
  complexity **without oscillation** — the one regime where monotone depth
  separation could exist, and exactly where Deep Lattice Networks (You et al.
  2017) claimed a depth benefit. A null here is the strongest form of the
  **H-strong** hypothesis.

The experiment measures Δ(c) = MSE(shallow) − MSE(deep) per family, so **positive
Δ means depth helps**. The reading: flat-at-zero across families ⇒ **H-strong**
(depth is intrinsically useless for constrained monotone nets); rising positive
with `c` (esp. teachers) ⇒ **H-weak** (depth doesn't help low-interaction
targets but does help as compositional complexity grows).

See the [design spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-07-12-monotone-depth-synthetic-probe-design.md)
for the full argument and scope.

## Results

```{note}
The probe machinery (Task 1–3) is complete and smoke-tested. The real numbers
below are filled by the GPU session per `benchmarks/RUNBOOK-depth-probe.md`:
sweep families {additive, teacher, lattice} × c ∈ {1, 2, 4, 8} with the fixed
deep/shallow bands (both GPUs, one (kind,c) per slot), then the iso-parameter
frontier only for families/c showing a signal. The plot and table are
committed after the GPU run.
```

(Results to be filled in by GPU session; see `benchmarks/RUNBOOK-depth-probe.md` for the run procedure.)
