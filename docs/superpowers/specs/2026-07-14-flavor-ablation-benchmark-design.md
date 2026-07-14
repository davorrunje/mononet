# Flavor ablation benchmark (mixed · alternate · split) — Design

**Date:** 2026-07-14
**Author:** Davor Runje
**Status:** Draft (design); open items pinned 2026-07-14. **Phase 1 merged (#108); Phase 2 CI-green (#109).** Executable once #109 merges.
**Parent:** §8 of [`2026-07-13-monotone-constructions-init-and-ablation-design.md`](2026-07-13-monotone-constructions-init-and-ablation-design.md). This doc is the concrete harness/study design for that ablation.
**Depends on:** phase 1 (mode rename `mixed`/`split`, merged) + phase 2 (`alternate` mode + composition-aware init via `prev=`, PR #109). The Phase-4 harness build requires #109 on `main`; the plan branch should rebase on `main` after #109 merges.

**Final phase-2 API this ablation targets (all three backends):** `MonoLinear`/`MonoDense(units, mode="alternate", activation=…, prev=<previous layer | None>)`. `prev=None` ⇒ entry (convex, `m_in=0`); each subsequent layer alternates phase and applies the composition-aware init from `prev`'s stored moment. `convex_fraction`/`init` are rejected under `alternate`; `MonoResidual` rejects `mode="alternate"` (build a custom `F` of `prev=`-chained layers instead).
**Built on:** the Stage-2 unified depth benchmark infra (`benchmarks/_common/{search,model_builder,config,runner,search_spaces}.py`, `benchmarks/search.py`) — [`2026-07-13-stage2-unified-depth-benchmark-design.md`](2026-07-13-stage2-unified-depth-benchmark-design.md).

## 1. Goal & hypotheses under test

Turn the frozen CPU findings (parent §6) into a reproducible GPU verdict on real data:

- **H-plain (headline):** `alternate` + composition-aware init trains deep *plain* monotone stacks where `mixed` and `split` diverge. Does the depth-16 divergence gap hold on real data, and does deep `alternate` translate into an accuracy/robustness win?
- **H-init (the ELU question):** for each activation, does the composition-aware init beat the *legacy* per-layer init for `alternate`? Expectation: relu/softplus collapse under legacy (documented on real data); elu/selu survive under both (parent §6d/§6e).
- **H-residual:** in `MonoResidual`, the near-identity start tames all three flavors, so alternation is a wash (parent §5.4) — confirm on real data.

## 2. Factor matrix

| factor | levels |
|---|---|
| **flavor** | mixed · alternate · split |
| **activation** | relu · elu · softplus · selu |
| **topology** | **plain (primary)** · residual (reference) |
| **depth** | 4 · 8 · 16 (plain layers / residual blocks) |
| **init (alternate only)** | **composition-aware** *and* **legacy** per-layer init (collapse baseline — user decision 2026-07-14) |

`mixed` uses `absolute_init_params`; `split` uses standard init; `alternate` is swept over **both** inits. The legacy-init `alternate` arm is the collapse baseline: it documents on real data *why* the composition-aware init is needed (relu/softplus expected to collapse; elu/selu expected to survive).

**Arm count per dataset:** mixed = 2 topo × 3 depth × 4 act = 24; split = 24; alternate = 2 topo × 3 depth × 4 act × 2 init = 48 → **96 configs/dataset** (before seeds/LR). Focused-first (§4) trims topology to plain to cut this ~2×.

## 3. Metrics (robustness-first)

Reuse Stage-2's primary metric + add the stability readout:
- **Primary:** ROC-AUC (classification) / MSE (regression) + seed-bootstrap CI.
- **Dispersion:** IQR of the primary metric across seeds.
- **Convergence:** epochs-to-best (early-stop epoch), median + IQR.
- **Divergence-rate (new, first-class):** fraction of `(seed, LR)` runs that diverge (loss > threshold / NaN). This is the headline for H-plain — divergence is measured, not footnoted.
- **LR mini-sweep** on 2 representative datasets (one classification, one regression): fix `(depth, width)`, vary LR on a grid, plot metric-vs-LR per flavor/init — the direct robustness readout.

## 4. Dataset scope (focused-first — user decision 2026-07-14)

**First run (cheap signal):**
- `heart` (binary classification, small, well-behaved) and `auto` (regression, small) — from the paper-5.
- The **synthetic complexity ladder** restricted to the `lattice` family × complexity {low, mid, high} — the cleanest monotone-nonlinear, depth-relevant target (Stage-2 §3); the most informative for H-plain.

**Expansion (if the focused run is informative):** the full Stage-2 set (10 real + full synthetic ladder), and the residual topology arm.

## 5. Harness integration (the new code, on top of Stage-2)

**Architecture decision (2026-07-14): a standalone fixed-architecture grid sweep, not the Optuna flavor-search path.** The existing `run_dataset`/`_ALL_FLAVORS`/`flavor_name`/`_parse_flavors`/`suggest_config` machinery Optuna-*searches* hyperparameters (including depth) per flavor. A flavor *ablation* instead needs each cell trained at a **fixed** architecture (depth/width/LR held constant across flavors) so the flavor effect is isolated rather than confounded with per-cell tuning budget. So Phase 4 adds a **new grid runner** `benchmarks/flavor_ablation.py` that enumerates cells `(dataset, flavor, activation, depth[, alt_init])`, builds a `BenchmarkConfig` per cell at fixed width/LR/seeds, calls the existing `runner.run(cfg, bundle)` (which loops seeds), and aggregates. This **reuses** `build_model`, `runner.run`, `_score_predictions`, the dataset registry, and the dual-GPU launcher pattern — and does **not** extend `_ALL_FLAVORS`/`flavor_name`/`_parse_flavors`/`suggest_config` (those stay the Optuna path for Stage-2). The LR mini-sweep is the same grid runner with the LR axis expanded on 2 datasets.

1. **Flavor identity in the grid runner.** A cell is `(mode, alt_init)` where `mode ∈ {"mixed","split","alternate"}` and `alt_init ∈ {None, "composition", "legacy"}` (non-None only for `alternate`). The grid runner names results `f"{mode}"`/`f"alt-{alt_init}"` — no change to the Optuna `flavor_name`.
2. **`model_builder` — build the alternate stack.** Today `_build_{torch,jax}_stack` / the keras inline stack loop with a scalar `convex_fraction`. Add an `alternate` branch keyed on `cfg.alt_init`:
   - **`alt_init="composition"`** (the real construction): thread `prev=` (phase-2 API) — entry `prev=None`, each subsequent `MonoLinear(mode="alternate", activation=…, prev=<previous>)`; identity head unchanged.
   - **`alt_init="legacy"`** (collapse baseline — **no new layer code**): build the *same pure-class* layers as `mode="mixed"` with `convex_fraction` alternating `1.0, 0.0, 1.0, …` per depth index. This is the identical `|W|` forward but with the *legacy* per-layer `absolute_init_params` init — i.e. the naive alternation that collapses for relu/softplus (parent §4.5). The two arms share the forward and differ only in init, which is exactly the comparison H-init wants.
   Residual alternate (`composition`): custom `F=[convex, concave]` per block, threaded with `prev=`, last layer near-zero-scaled (parent §5.3).
3. **Config — `benchmarks/_common/config.py`.** Add `"alternate"` to the `mode` Literal and add `alt_init: Literal["composition", "legacy"] | None = None` (None for non-alternate). Both round-trip in `config_io`.
4. **Pre-check gate (new).** Before training each `(flavor, activation)` cell, a fast init-time screen: assert non-zero output variance + non-vanishing gradient (catch collapse) and monotonicity (finite-difference). A collapsing cell is `log`-ged and recorded as `collapsed=True` in results (so the legacy-init relu/softplus collapse is *data*, not a crash); it is not silently dropped.
5. **Divergence-rate — `benchmarks/_common/runner.py` + reporting.** Record per-run divergence (final loss > threshold or NaN); aggregate the fraction over `(seed, LR)` into the result schema and the tables.

## 6. Outputs → `docs/benchmarks/`

Three studies (parent §8):
1. **Flavor study** — mixed vs alternate vs split × activation × topology × depth: primary + dispersion + convergence.
2. **Initialization study** — per flavor: divergence-rate vs depth; the alternate composition-vs-legacy comparison per activation (the collapse baseline made visible); split's deep-init behavior.
3. **Residual study** — each flavor in `MonoResidual` vs its plain counterpart at matched depth/params (H-residual).

## 7. Orchestration

Dual-GPU via the Stage-2 `stage2_launch.py` process-pool pattern (one **dataset** per subprocess, round-robin `--devices cuda:0 cuda:1`, work-stealing `ThreadPoolExecutor`). Each subprocess runs the grid runner for its dataset over all flavor cells; the grid runner is single-threaded per process (no Optuna, so no `n_jobs` deadlock concern, but keep one process per GPU). No merge step needed — each writes its own `benchmarks/results/flavor-ablation/<dataset>.json` (one record per cell). JSON results are the committed artifact.

## 8. Sequencing & dependency

1. **After phase 1 lands** (rename): the harness already speaks `mixed`/`split`.
2. **After phase 2 lands** (`alternate` + `prev=` + composition-aware init): implement §5 (flavors, builder `prev=` wiring, `alt_init` config, pre-check gate, divergence metric) with CI-tested unit/smoke tests — one PR-ready branch. Build against the real phase-2 API.
3. **Focused run** (§4) on GPU; commit results; read H-plain / H-init / H-residual.
4. **Expand** to the full Stage-2 set + residual if the focused run is informative; write the three benchmark docs.

Steps 2 is the code deliverable (its own plan, gated on phase 2); steps 3–4 are controller-run + write-up.

## 9. Pinned decisions (2026-07-14)

- **Flavor-key encoding:** extend the flavor identity to a 4-tuple `(mode, residual, deep, alt_init)` where `alt_init ∈ {None, "composition", "legacy"}` (None for `mixed`/`split`). `flavor_name` emits `mixed-plain` / `split-deep` unchanged for the two-init-less flavors, and `alt-comp-plain` / `alt-legacy-plain` / `alt-comp-deep` … for the alternate arms. Keep the existing 3-tuple entries working (treat a missing 4th field as `None`).
- **Divergence:** a run is `diverged` if its final validation loss is non-finite (NaN/inf) **or** `> 10 ×` the predict-the-mean baseline (`Var[y_val]` for regression / the majority-class log-loss for classification). Divergence short-circuits early-stopping (stop and mark). `divergence_rate` = fraction of `(seed, LR)` runs diverged, persisted per `(dataset, flavor, depth, activation)`.
- **LR grid** for the mini-sweep: `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`. **Mini-sweep datasets:** `heart` (classification) + `auto` (regression).
- **Legacy-init arm activations:** run all four (relu · elu · softplus · selu) so the collapse (relu/softplus) *and* survival (elu/selu) are both documented on real data — do not pre-trim.
- **Width/param matching:** report plain at `depth ∈ {4,8,16}` layers and residual at `blocks ∈ {2,4,8}` (= `{4,8,16}` monotone layers, since each block = 2 layers); the docs tables compare plain vs residual at equal *layer* count, and state the block↔layer mapping.
- **Focused-first run** (the first GPU pass): `{heart, auto}` + synthetic `lattice` × {low,mid,high}, **plain topology only**, all flavors/activations/depths + both alt inits. Expand to the full Stage-2 set + residual arm only if the focused pass is informative.
