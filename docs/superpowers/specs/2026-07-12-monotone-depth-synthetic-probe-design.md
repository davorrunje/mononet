# Does depth ever help constrained monotone networks? A synthetic probe

Status: Draft (design) — awaiting review

Benchmark-only (no `mononet` package/kernel change; `benchmarks/` stays out of
the wheel). Theoretical follow-up to the depth-null empirical result — see
[[depth-null-in-constrained-monotone-nets]], the loan size-ladder
([design](2026-07-10-loan-size-ladder-design.md)), and the large-dataset screen
([design](2026-07-11-large-dataset-screen-design.md)).

## 1. Problem / motivation

Across the loan size-ladder (no dose-response in N) and the 5-dataset max-size
screen, deep `absolute`-residual monotone stacks never beat shallow ones
(Δ = IQM(deep) − IQM(shallow) ≈ 0 or negative everywhere; every dataset earned
the `standard` gate verdict). This is a robust empirical null, but the real
datasets cannot tell us **why**, because they conflate three possible causes:

- **(E) Expressivity/prior.** The CMNN construction is a universal approximator
  of monotone functions at shallow depth (Runje & Shankaranarayana 2023; Sartor
  et al. 2025 on bounded-activation limits). Depth-separation theorems for
  unconstrained ReLU nets (Telgarsky 2016 sawtooth; Eldan & Shamir 2016; Montúfar
  et al. 2014 linear-region counting) all rely on **oscillatory folding** to
  multiply complexity with depth — a mechanism **monotonicity forbids**. So depth
  may add no *representable* function the monotone target needs.
- **(D) Data simplicity.** Credit/income targets are near-additive (low-order
  interactions); even if depth helped complex monotone targets, these give it
  nothing to exploit.
- (O) **Optimization.** Constrained (non-negative-weight) deep stacks may be
  hard to train — we observed **collapse to the majority base rate** on taiwan
  and lc (both arms). Residual connections prevent degradation but create no
  depth *advantage*.

**This probe separates them** using synthetic *monotone* targets with a
controllable complexity knob, so we can state the theorem precisely:

- **H-strong:** the constrained-monotone class is *depth-insensitive* — depth
  never helps, even for complex/compositional monotone targets.
- **H-weak:** depth doesn't help *low-interaction* monotone targets (the tabular
  case) but does help as monotone-compositional complexity grows.

Either outcome is a useful result: H-strong is a clean "depth is not a useful
axis for constrained monotone nets" statement (and a sharp contrast with Deep
Lattice Networks, You et al. 2017, which claimed depth benefits for stacked
monotone lattices); H-weak refines the null to "data-driven, not intrinsic."

## 2. Goals & constraints

- Reuse the existing `benchmarks/` harness: the `run_ladder` deep/shallow arms,
  `search_spaces`, `delta_by_n`. Add only a synthetic-target generator, a probe
  run script, and a report.
- **No `mononet` package/kernel change.** The monotone *teacher* (below) is
  built from `mononet.core.reference` (the NumPy ground-truth), not new layers.
- Frame every result as **Δ(c)** — the deep−shallow gap as a function of the
  complexity knob `c` — directly comparable to the screen's Δ.
- Isolate expressivity/data (E/D) from optimization (O) by design (§3.3).

## 3. Experiment design

### 3.1 Synthetic monotone targets (all monotone ↑ in every input; domain [0,1]^d, d = 6)

Monotonicity of each target is asserted numerically (coordinate-wise finite
differences ≥ 0 on a dense grid) at generation time. Complexity knob `c`.

1. **Additive control** (`c` ignored): f = Σᵢ gᵢ(xᵢ), gᵢ random monotone
   piecewise-linear. Depth must not help — anchors the null / sanity check.
2. **Teacher-depth sweep** (the acid test): the target *is* a random **monotone
   teacher network** of depth `c ∈ {1, 2, 4, 8}`, built from the reference
   construction (non-negative weights + the convex/concave activation split).
   Guarantees the target is monotone *and* representable by a deep student of
   matching size — so a persistent deep-student error is **optimization (O)**,
   while a shallow-only failure is **expressivity (E)**.
3. **Max/min-lattice teacher** (most depth-favoring): f = nested max/min of
   monotone terms, `c` = nesting depth. Max/min composition builds piecewise
   complexity **without oscillation** — the one regime where monotone
   depth-separation could exist, and exactly where Deep Lattice Networks claimed
   a depth benefit. A null here is the strongest form of H-strong.

(Optional, deferred: an **interaction-order** polynomial family — Σ over random
`c`-subsets of ∏ xᵢ — testing pure interaction capacity. Nice-to-have; families
1+2+3 already separate H-strong/H-weak.)

### 3.2 Students / arms

Identical to the screen: `absolute` `MonoResidual`, **deep** D ∈ {6, 10, 16} vs
**shallow** D ∈ {1, 4}, same `search_spaces` (large-batch band — the synthetic
sets are dense/large). Metric: **MSE** (regression) — noise-free, so this is a
clean *approximation* question ("can it fit the monotone target?"), sidestepping
the accuracy-ceiling/collapse artifact that muddied the imbalanced classification
sets. A held-out test split confirms it is fitting, not memorizing.

### 3.3 Controls that give the answer meaning

- **Noise-free, large-sample** (dense [0,1]^6 sampling): an approximation test,
  not a generalization test.
- **Teacher-student representability guarantee** (families 2–3): a deep student
  of the teacher's size can represent the target *exactly*, so the deep-student
  error cleanly isolates **O** from **E**.
- **Iso-parameter frontier** (the rigorous depth control): depth-separation is a
  *parameter-efficiency* claim, so for each `c` also sweep (depth, width) along an
  equal-parameter-count curve and locate the error minimum. If it sits at high
  width rather than high depth, depth is genuinely not the useful axis. (Run this
  only for families/`c` where the fixed deep/shallow bands show a signal — see
  §6.)

### 3.4 Readouts

- **Δ(c)** per family — the headline. **Sign convention:** since the metric is
  MSE (lower is better), define Δ = MSE(shallow) − MSE(deep), so **positive Δ
  means depth helps** (matching the screen's "positive ⇒ depth better"). Report
  the seed-bootstrap band as in the screen. Flat-at-zero across families ⇒
  H-strong; rising positive with `c` (esp. families 2–3) ⇒ H-weak.
- **Teacher-student recovery gap** — deep-student MSE when the target is a deep
  monotone teacher; paired with a collapse count (the O signal).
- **Iso-parameter frontier** — MSE vs depth-fraction at fixed params, per `c`.

### 3.5 Pre-registered predictions

- Family 1: Δ ≈ 0 (sanity).
- Family 2: E + the CMNN UAT predict Δ ≈ 0 even as teacher depth grows; a rising
  Δ would falsify H-strong and support H-weak.
- Family 3: the decisive test vs Deep Lattice Networks — Δ ≈ 0 ⇒ strong result.

## 4. Architecture / components

- **`benchmarks/datasets/synthetic.py`** — `synth_monotone(kind, c, *, d=6,
  n_train, n_test, seed) -> DatasetBundle` (task `"regression"`, all features
  `mono_increasing`, `X ~ U[0,1]^d`, `y = f(X)`). `kind ∈ {additive, teacher,
  lattice}`. Includes the numerical-monotonicity assertion.
- **Monotone teacher** — a small helper (in the same module) composing
  `mononet.core.reference` monotone-dense/residual ops with fixed non-negative
  weights (seeded) to define families 2–3. No new package code.
- **`benchmarks/monotone_depth_probe_run.py`** — for each `(kind, c)`, wrap the
  synthetic bundle and run the existing `run_ladder` deep/shallow arms (MSE),
  then `delta_by_n`-style Δ; emit `benchmarks/results/depth-probe/<kind>.json`
  (records keyed by `c`).
- **`benchmarks/_common/depth_probe_report.py`** — Δ(c) table + plot per family
  (PNG + PDF, mirroring `size_ladder_report`), plus the iso-parameter-frontier
  plot.
- **`docs/benchmarks/monotone-depth-probe.md`** — results page (skeleton now;
  filled by the GPU run) + the theory framing (§1) tying it to the screen and the
  two source papers.
- **`benchmarks/RUNBOOK-depth-probe.md`** — GPU run + report procedure.

## 5. Testing

- **Generator unit tests**: each `kind` yields the declared shape, `y` finite,
  every feature index in `mono_increasing`, and the **numerical monotonicity
  assertion passes** (coordinate-wise ≥ 0); teacher/lattice targets are exactly
  reproducible per seed.
- **Probe smoke test**: tiny synthetic bundle, both arms, ≤2 trials/1 seed →
  a record with a finite Δ per `c`. Fast, CI-cheap.
- **Report**: Δ(c) table + `render` writes PNG **and** PDF; iso-param plot smoke.
- Existing suites green; `uv run mypy` clean; `sphinx-build -W` clean.

## 6. Scope split

- **Landed in this PR (plumbing + smoke):** the generator + tests, the probe run
  script + smoke, the report, the docs skeleton, the RUNBOOK. Mergeable with no
  real numbers.
- **GPU session (per RUNBOOK):** the real probe — families 1, 2, 3 with the
  fixed deep/shallow bands first (quick Δ(c) signal); then the **iso-parameter
  frontier** only for families/`c` that show a signal. Fills the report + docs.

## 7. Non-goals / out of scope

- A formal proof of monotone depth-separation (or its absence) — this probe is
  the empirical/constructive evidence that would *motivate* such a proof.
- `switch`-mode or other-backend sweeps (torch `absolute` only, matching the
  screen), and real-data work (covered by the screen).
- Any `mononet` package/kernel change.

## 8. Open items (for review)

1. **Family set.** Proposed: **1 (control) + 2 (teacher-depth) + 3 (max/min
   lattice)** — the pair that separates H-strong/H-weak and gives the DLN
   contrast. The interaction-order polynomial family is deferred as nice-to-have.
   Confirm or adjust.
2. **Iso-parameter frontier vs fixed bands.** Proposed: fixed deep/shallow bands
   first (cheap, directly comparable to the screen), frontier only where a signal
   appears. Confirm or run the frontier everywhere.
3. **Dimension / sample size.** `d = 6`, dense sampling — enough that complexity
   comes from interactions, not dimensionality. Confirm `d`.
4. Whether to also re-score the *real* screen datasets with **AUC/PR** (to rule
   out the metric-ceiling caveat) as part of this write-up or separately.
