# Stage-2: Unified Depth Benchmark (all datasets + synthetic, post-fix re-run) — Design

**Date:** 2026-07-13
**Status:** Draft (design)
**Sub-project:** B/C (depth evaluation after the MonoResidual gate fix, #100).
**Package area:** `benchmarks/` (runners, search space, registry, gate, ladder) + `docs/`. No
`mononet/` package change.
**References:** #100 (near-zero init + softplus gate — the fix this re-evaluates); the retired
#90 large-dataset screen and #99 synthetic depth probe (folded in here); the loan size-ladder
([2026-07-10-loan-size-ladder-design.md](2026-07-10-loan-size-ladder-design.md)); the depth
theory ([[depth-null-in-constrained-monotone-nets]]).

> **Goal.** Re-evaluate whether depth helps constrained monotone networks, now that #100 makes
> the residual branch actually engage. Consolidate the three overlapping depth vehicles (main
> benchmark, #90 screen, #99 probe) into **one** run / results set / PR / doc: a two-stage study
> — **Stage A** screens deep-vs-shallow across all datasets (10 real + a synthetic complexity
> ladder) with a significance gate, and **Stage B** size-ladders the large real datasets that
> pass. No assumption that deep wins anywhere: on monotone-constrained nets, a deep advantage
> even on depth-constructed targets is exactly the open question, so it is measured, not asserted.

## 1. Motivation & framing

Before #100 the residual branch `F` never engaged (two traps), so every prior depth result was
confounded — "deep is neutral" (#90) could not distinguish "depth doesn't help" from "depth was
never used." #100 fixed the mechanism (gate opens, `F` trains). This spec produces the *trustworthy*
depth verdict on the fixed layer.

**Key scientific stance (informs the whole design).** Classical depth-separation (Telgarsky
sawtooth, etc.) relies on oscillatory folding that **monotonicity forbids**, so there is *no
guarantee* a deep monotone net beats a shallow one — even on a target *constructed* to reward
depth. Therefore synthetic targets are **empirical measurements on the same footing as real data**,
not pass/fail validations. The synthetic complexity sweep is how we measure *whether, and how, a
deep advantage emerges as target complexity grows*.

## 2. Consolidation (what happens to #90 and #99)

One unified benchmark. The main benchmark (`benchmarks/search.py` → `deep-residual-accuracy`)
**becomes** the screen:

- **#90 (large-dataset screen)** — *retired*. Its distinct value was the significance gate
  (`benchmarks/_common/screen_gate.py`) and the 5 large-dataset onboardings (`adult, taiwan,
  polish, german, lc`). The gate is ported into Stage A's analysis; the datasets are wired into
  the main benchmark. The screen's separate runner/launcher/doc are superseded. Close PR #90 as
  superseded (its dataset-onboarding commits are already on `main`).
- **#99 (synthetic depth probe)** — *folded in*. Its generator (`benchmarks/datasets/synthetic.py`,
  not yet on `main`) is ported (fixing the `teacher_elu ≡ teacher_relu` dup bug), and synthetic
  targets become benchmark rows via a generator-backed registry source. The probe's dedicated
  runner/report/doc are superseded. Close/redirect PR #99.

Result: one runner, one results tree, one table, one PR, one doc.

## 3. Datasets

Wire all into `benchmarks/search.py::_ALL_DATASETS` and the registry.

**Real (10).** Paper 5 (`auto`, `heart`, `compas`, `loan`, `blog`) + #90's `adult`, `taiwan`,
`polish`, `german`, `lc`. Already CSV/LFS-backed and in `DATASETS_SPEC`.

**Synthetic complexity ladder (generated).** Teacher families {`additive`, `teacher_relu`,
`teacher_elu`, `lattice`} × complexity `c ∈ {low, mid, high}` (12 rows), each a fixed size
(e.g. `n_train=16000`, `n_test=4000`) generated deterministically from a seed. Named
`synth_<family>_c<level>`. This is the controlled measurement of depth-advantage vs target
complexity.

**The targets MUST be genuinely nonlinear (non-degeneracy requirement).** The #99 generator
is degenerate under a `[0,1]`-input / non-negative-weight monotone construction: every ReLU/ELU
preactivation stays strongly positive, so `act(z)=z`, every layer is affine, and the whole
teacher collapses to a **linear** function (empirically `teacher_relu`/`teacher_elu` R²=1.00000
against a linear fit *and byte-identical to each other*; `additive` R²≈0.98, `lattice` R²≈0.93).
A near-linear target is trivially shallow-learnable, so "deep doesn't win" would be a
target-linearity artifact, not evidence about monotone depth. The fixed generator must:
- **Center the teacher's input sampling** (e.g. `x ~ U[−1,1]` or standard-normal) so ReLU/ELU
  preactivations straddle 0 and the nonlinearity bites — this makes `teacher_relu` (sharp) and
  `teacher_elu` (smooth) genuinely nonlinear *and genuinely different from each other*.
  Monotonicity is preserved (non-negative weights + monotone activation hold for any input range;
  all synthetic features remain non-decreasing).
- **Strengthen `lattice`** (nested min/max of monotone single-hidden-layer ReLU experts — pure
  min/max of *affine* terms plateaus at R²≈0.9 on a bounded box and cannot clear the gate, so the
  experts are nonlinear-but-monotone) — the cleanest monotone-nonlinear, depth-relevant family,
  non-degenerate regardless of input sign; deepen the
  min/max nesting so it is strongly nonlinear (the current R²≈0.93 is too shallow).
- **Spread `additive` knots** across the (centered) input range so the ramps actually fire.
- **Acceptance gate:** a committed test asserts each family at high `c` fits a linear model with
  **R² < 0.7**, so a degenerate target can never ship again.

**Registry change (generator-backed source).** `DatasetSpec` / `benchmarks/datasets/registry.py`
/ `loader.py` currently resolve datasets from committed CSVs. Add a **generator source**: a spec
variant that carries a callable (family, c, sizes, seed) → `DatasetBundle` instead of a CSV path,
so `search.py`/`run_dataset` pull `synth_*` through the same interface as real datasets.

## 4. Metric (ROC-AUC primary for classification)

Accuracy is base-rate-trivial on the imbalanced datasets (loan 0.649 ≈ base rate 0.647; polish
~4.7% positive) and hides depth effects. Change:

- Extend the metrics vocabulary (`benchmarks/_common/config.py` `metrics` Literal) with
  `"roc_auc"` (and keep `accuracy`, `mse`, `rmse`).
- Compute ROC-AUC in `benchmarks/_common/runner.py` (needs predicted probabilities +
  `sklearn.metrics.roc_auc_score`); still compute accuracy for the table.
- `benchmarks/_common/search.py::_primary_metric` returns `"roc_auc"` for
  `binary_classification`, `"mse"`/`"rmse"` for regression. This becomes the objective's primary
  metric **and** the gate metric. Accuracy remains a reported secondary column.

Regression datasets (`auto`, `blog`, synthetic) rank/gate on MSE/RMSE as today.

## 5. Stage A — screen (all datasets)

Per `(dataset, flavor)`: Optuna-tuned search + final eval, exactly the current `run_dataset`
pipeline, extended to the new datasets/metric.

- **Flavors (6):** `{switch, absolute} × {plain, residual, deep}`.
- **Budgets (per-dataset defaults in `run_dataset`).** Small datasets → more trials / more seeds /
  k-fold CV; large datasets (`n_train ≥ 20_000`: `loan, lc, adult, taiwan, blog`) → fewer trials /
  holdout + the large-batch band (#100's size-driven `search_spaces`). New large datasets reuse
  the large preset; new small (`german`, `polish`, `compas`) reuse the small preset. Synthetic
  rows use a fixed cheap preset (they're fast).
- **Significance gate.** Stage A always **emits the raw evidence** per dataset — Δ =
  (best-deep − best-shallow) point estimate + a seed-bootstrap 95% CI (Δ_lo, Δ_hi), in the
  dataset's primary metric — so the gate can be applied *after* the run. The gate
  (`benchmarks/_common/screen_gate.py`, generalized from accuracy-only to **metric-aware**) is a
  pure function `gate(delta_lo, delta_point, margin, direction) → verdict`; its **practical-margin
  floor is a parameter decided post-results** by inspecting the observed Δ distribution (per
  decision 2026-07-13), *not* hardcoded. The significance component (`Δ_lo > 0` in the metric's
  improvement direction) is objective and reported regardless. Verdict per dataset:
  **deep-better** (significant *and* Δ_point ≥ chosen margin), **neutral**, or **deep-worse**.
  "best-shallow" = better of the 4 non-deep flavors; "best-deep" = better of the 2 deep flavors,
  by primary metric. Because the margin is chosen after Stage A, the Stage-A → Stage-B gating is
  an explicit analysis step, not an in-run automatic route.
- **Output.** One results tree `benchmarks/results/stage2/<dataset>-<flavor>.json` (schema =
  current `run_dataset` output + `roc_auc`), consumed by the docs table and the gate.

## 6. Stage B — size-ladder (gated large real datasets only)

For each **large real** dataset with a **deep-better** Stage-A verdict, run the size-ladder
(generalize `benchmarks/loan_ladder_launch.py` / `loan_size_ladder_run.py` beyond `loan`): deep
vs shallow, each **tuned independently** at each training-set subsample size, IQM + seed-bootstrap
CI on Δ(size). Answers "is the edge real and does it grow with scale, or vanish in noise?" —
the loan-ladder finding, now applied to every gated large dataset.

- **Small datasets cannot be size-laddered** (too few rows). A small dataset with a Stage-A
  deep-better verdict gets a "re-run with more seeds" note, not a ladder.
- **Synthetic** keeps its complexity sweep in Stage A. Size-laddering synthetic (vary `n`) is a
  documented optional follow-up, not in this spec.
- If Stage A gates **zero** large datasets to ladder, Stage B is skipped and the verdict stands on
  Stage A — a legitimate (and, given the depth-null prior, likely) outcome; `log`/document it.

## 7. Orchestration

Dual-GPU (5090 + 3090) via the `benchmarks/screen_launch.py` process-pool pattern: one
`(dataset)` per subprocess, `n_jobs=1` inside each (threaded Optuna deadlocks under process
nesting — established), round-robin over `cuda:0,cuda:1` (repeat a device in the pool to co-locate
cheap datasets). Optuna storage = resumable `*.db` under a git-ignored `studies/` dir; JSON results
are the committed artifact. Stage A is the dominant cost; Stage B runs only on the few winners.
The launcher already hardcodes `n_jobs=1`; extend it (or add a thin Stage-A launcher) to cover the
6-flavor main-benchmark run across the pool.

## 8. Docs

- **`docs/benchmarks/deep-residual-accuracy.md`** — the unified table: 10 real + 12 synthetic rows,
  per dataset `best-shallow (mode) / best-deep (mode) / Δ / gate verdict / deep depth`, primary
  metric = ROC-AUC (classification) / MSE-RMSE (regression), accuracy reported alongside. Synthetic
  rows grouped in their own sub-table (family × c) with the depth-vs-complexity reading.
- **`docs/concepts/monotonic-residual.md`** — fill the "Depth on real data (before/after)"
  `{note}` placeholder (left in #100) with the Stage-A/B verdict.
- **`docs/benchmarks/`** — Stage-B ladder page(s) for any gated dataset (generalize the loan
  size-ladder doc); if none gate, a one-paragraph "no dataset advanced to a ladder" result.
- **README** — refresh/extend the benchmark table (or keep README to a summary + link to the full
  page — decide in the plan to avoid an unwieldy README table).
- Retire the `large-dataset-screen` doc page (superseded); update the depth-probe page or fold it
  into the synthetic sub-table.

## 9. Components / repo layout

```
benchmarks/search.py                     # _ALL_DATASETS += 5 real + 12 synth; wire budgets/metric
benchmarks/_common/config.py             # metrics Literal += "roc_auc"
benchmarks/_common/runner.py             # compute roc_auc (probs + sklearn); keep accuracy
benchmarks/_common/search.py             # _primary_metric -> roc_auc for classification
benchmarks/_common/screen_gate.py        # metric-aware margins + gate verdict (extend)
benchmarks/datasets/{spec,registry,loader}.py  # generator-backed source for synth_*
benchmarks/datasets/synthetic.py         # NEW on main (port from #99); fix teacher_elu switch
benchmarks/stage2_launch.py              # NEW (or extend screen_launch): dual-GPU Stage-A pool
benchmarks/size_ladder_run.py + launch   # generalize loan_ladder_* to any large dataset (Stage B)
benchmarks/results/stage2/*.json         # committed Stage-A results
benchmarks/results/size-ladder/<ds>/*.json  # committed Stage-B results (gated datasets)
tests/benchmarks/…                        # roc_auc metric, metric-aware gate, generator source,
                                          # synth teacher_elu != teacher_relu, launcher round-robin
docs/benchmarks/deep-residual-accuracy.md, docs/concepts/monotonic-residual.md, README.md
```

## 10. Testing

Fast, deterministic unit/smoke tests (the heavy runs are manual/controller, committed with
results — CI never runs them):

- `roc_auc` scoring in `runner.py` (known probs → known AUC); `_primary_metric` returns `roc_auc`
  for a classification bundle, `mse`/`rmse` for regression.
- Metric-aware gate: `deep-better` only when `Δ_lo > 0` **and** `Δ_point ≥ margin` for each metric;
  boundary cases per metric direction (higher-better AUC vs lower-better MSE).
- Generator source: `load("synth_teacher_relu_cmid")` returns a monotone `DatasetBundle`;
  determinism (same seed → same data).
- **Non-degeneracy gate:** each family at high `c` fits a linear model with **R² < 0.7** (guards
  against the linear-collapse found in #99); `teacher_elu` targets **differ** from `teacher_relu`
  (`max|Δy| > 0` on centered inputs); every synthetic target is monotone non-decreasing in each
  feature.
- Launcher: datasets distributed round-robin over devices, `n_jobs=1` passed to each subprocess.
- `search.py --smoke --dry-run` covers all datasets incl. `synth_*` without running training.
- Full green: `pre-commit`, strict mypy (`--group bench`), ruff, docs build (`-W`).

## 11. Non-goals

- No `mononet/` package change (this is benchmarks + docs only).
- No new metric beyond ROC-AUC for classification (no F1/calibration/etc. — YAGNI).
- No size-laddering of synthetic targets (complexity sweep in Stage A suffices; optional later).
- Not re-running #90/#99 as separate artifacts — they are consolidated, not duplicated.
- No auto-merge of the follow-up PR; results land as a reviewed PR like #100.

## 12. Staged plan / sequencing

1. **Infra (code, CI-tested):** roc_auc metric + `_primary_metric`; metric-aware gate; generator
   registry source + `synthetic.py` (teacher_elu fix); wire 15 datasets into `search.py`; Stage-A
   launcher; generalized size-ladder runner. All with unit/smoke tests, one PR-ready branch.
2. **Stage A run (manual, dual-GPU):** the 6-flavor screen across 10 real + 12 synthetic; commit
   `benchmarks/results/stage2/*.json`; compute gate verdicts.
3. **Stage B run (manual):** size-ladder the gated large datasets (if any); commit results.
4. **Docs + PR:** fill the unified table + before/after + ladder pages; refresh README; retire the
   screen page; open the follow-up PR; close #90 (superseded) and #99 (folded in).

Infra (step 1) is the code deliverable this spec's plan covers; steps 2–4 are controller-run
execution + write-up gated on step 1 landing.

## 13. Open items

- Practical-margin floors per metric: **deferred to post-Stage-A analysis** (decided from the
  observed Δ distribution — decision 2026-07-13). The plan builds the gate to take `margin` as a
  parameter and Stage A to emit Δ + bootstrap CI unconditionally; no default margin is baked in.
  The gate's unit tests pin *behavior* (significant-and-≥-margin logic, per direction), not a
  specific margin value.
- Synthetic `c` levels: concrete definitions of low/mid/high per family (depth of the teacher /
  number of interacting features) — pin in the plan; must produce genuinely monotone targets
  (the additive/ReLU-ramp lesson from #99's Task 1).
- README table size: full 22-row table vs summary+link — decide in the plan.
- Budget presets for the 3 new small datasets (`german`, `polish`, `compas` already sized) and the
  synthetic rows — pin exact trials/seeds/CV in the plan.
