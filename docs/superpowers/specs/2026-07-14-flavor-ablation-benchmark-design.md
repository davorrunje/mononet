# Flavor ablation benchmark (mixed · alternate · split) — Design

**Date:** 2026-07-14
**Author:** Davor Runje
**Status:** Draft (design); **executable once phases 1–2 land**.
**Parent:** §8 of [`2026-07-13-monotone-constructions-init-and-ablation-design.md`](2026-07-13-monotone-constructions-init-and-ablation-design.md). This doc is the concrete harness/study design for that ablation.
**Depends on:** phase 1 (mode rename `mixed`/`split`) + phase 2 (`alternate` mode + composition-aware init via `prev=`). Harness wiring can be built now against the phase-2 API; the run happens after both land.
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

1. **Flavors — `benchmarks/_common/search.py::_ALL_FLAVORS`.** Add `alternate` arms. Because `alternate` carries an init variant, the flavor identity is no longer just `(mode, residual, deep)`; extend the flavor key with an `init` tag for alternate (e.g. `alternate-comp`, `alternate-legacy`) or add a 4th tuple field. `flavor_name` emits e.g. `alternate-comp-plain`, `alternate-legacy-deep`. Pin the encoding in the plan (least-churn, keeps `flavor_name` stable).
2. **`model_builder` — build the alternate stack via `prev=` chaining.** Today `_build_{torch,jax}_stack` / the keras inline stack loop with a scalar `convex_fraction`. Add an `alternate` branch that constructs the stack by threading `prev=` (phase-2 API): entry layer `prev=None`, each subsequent `MonoLinear(mode="alternate", activation=…, prev=<previous>)`; identity head unchanged. A `use_legacy_init` flag selects the collapse-baseline init (build the same pure-class layers with the existing per-layer init instead of `prev=`). Residual alternate: custom `F=[convex, concave]` per block with the near-zero last layer (parent §5.3).
3. **Config — `benchmarks/_common/config.py`.** Add `alt_init: Literal["composition", "legacy"] | None` (None for non-alternate flavors). Round-trips like the other fields.
4. **Pre-check gate (new).** Before training each `(flavor, activation)` cell, a fast init-time screen: assert non-zero output variance + non-vanishing gradient (catch collapse) and monotonicity (finite-difference). A collapsing cell is `log`-ged and recorded as `collapsed=True` in results (so the legacy-init relu/softplus collapse is *data*, not a crash); it is not silently dropped.
5. **Divergence-rate — `benchmarks/_common/runner.py` + reporting.** Record per-run divergence (final loss > threshold or NaN); aggregate the fraction over `(seed, LR)` into the result schema and the tables.

## 6. Outputs → `docs/benchmarks/`

Three studies (parent §8):
1. **Flavor study** — mixed vs alternate vs split × activation × topology × depth: primary + dispersion + convergence.
2. **Initialization study** — per flavor: divergence-rate vs depth; the alternate composition-vs-legacy comparison per activation (the collapse baseline made visible); split's deep-init behavior.
3. **Residual study** — each flavor in `MonoResidual` vs its plain counterpart at matched depth/params (H-residual).

## 7. Orchestration

Dual-GPU via the Stage-2 `screen_launch.py` process-pool pattern (one dataset per subprocess, `n_jobs=1`, round-robin devices). Optuna storage resumable `*.db` under git-ignored `studies/`; JSON results are the committed artifact under `benchmarks/results/flavor-ablation/`.

## 8. Sequencing & dependency

1. **After phase 1 lands** (rename): the harness already speaks `mixed`/`split`.
2. **After phase 2 lands** (`alternate` + `prev=` + composition-aware init): implement §5 (flavors, builder `prev=` wiring, `alt_init` config, pre-check gate, divergence metric) with CI-tested unit/smoke tests — one PR-ready branch. Build against the real phase-2 API.
3. **Focused run** (§4) on GPU; commit results; read H-plain / H-init / H-residual.
4. **Expand** to the full Stage-2 set + residual if the focused run is informative; write the three benchmark docs.

Steps 2 is the code deliverable (its own plan, gated on phase 2); steps 3–4 are controller-run + write-up.

## 9. Open items

- Flavor-key encoding for the init variant (4th tuple field vs name tag) — pin in the plan.
- Divergence threshold + early-stop interaction — pin (e.g. final val loss > 10× the predict-mean baseline ⇒ diverged).
- LR grid + the 2 LR-mini-sweep datasets — pin.
- Whether the legacy-init alternate arm runs on all 4 activations or only relu/softplus (where collapse is expected) to save compute — decide from the focused run.
- Width/param-matching between plain (depth = layers) and residual (depth = blocks × 2 layers) so comparisons are at equal layer count (parent §6 bookkeeping).
