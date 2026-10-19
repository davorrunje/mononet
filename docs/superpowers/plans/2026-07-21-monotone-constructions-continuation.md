# Monotone constructions (mixed · alternate · split) — continuation / handoff

**Date:** 2026-07-21
**Purpose:** resume point for the `alternate`-construction program. Read this + the linked spec/plans, then pick up the outstanding work (§ "What's next").

## TL;DR

The `alternate` construction and its composition-aware `prev=` init are **built, merged, and CI-green** across all three backends. The **one experiment that actually tests why `alternate` exists — the deep *plain* verdict — has not been run.** Everything measured so far is either shallow (parity) or residual (a wash). Finish the deep-plain grid run before drawing conclusions.

## Where we are (all on `main`)

| Phase | PR | State |
|---|---|---|
| 1 — rename `mode` `absolute→mixed`, `switch→split` (hard break) | #108 | ✅ merged |
| 2 — `alternate` construction + composition-aware init via `prev=` (torch/jax/keras) | #109 | ✅ merged |
| 4 — flavor-ablation harness + plan (`benchmarks/flavor_ablation.py`, depths {4,8,16}, divergence-rate) | #110 | ✅ merged |
| — tuned **shallow** (≤4-layer) 4-flavor bake-off + `convex_fraction` ablation | #111 | ✅ merged |
| — HP-search sensitivity curves | #117 | ✅ merged |

**API (live):** `MonoLinear`/`MonoDense(units, mode="alternate", activation=…, prev=<prev|None>)`. `prev=None` ⇒ entry (convex, `m_in=0`); each layer alternates phase and applies the composition-aware init from `prev`'s stored moment. Layer-resolved to the `|W|` kernel (kernels/equivalence untouched). `convex_fraction` reserved for `mixed` (rejected under `alternate`); `MonoResidual` rejects `mode="alternate"` (build a custom `F` of `prev=`-chained layers).

## What the evidence says so far

- **Shallow tuned bake-off (#111, `docs/benchmarks/alternate-base-result.md`):** at ≤4 tuned layers on the paper-5 datasets, **`alternate` does not decisively beat `mixed`/`split`** — co-leads `compas`, parity on `auto`/`heart`, loses `blog`/`loan`. The biggest single win was **fixing `convex_fraction = 0.5`** in `mixed` (wins 3/5) → proposed default.
- **Deep *residual* screen (#115, DRAFT):** "monotone depth is **neutral** across all 5 datasets" — deep residual ≈ shallow. This is the **H-residual** regime (near-identity start tames all flavors) — consistent with the CPU prior that residual is a wash for alternation.
- **CPU exploration prior (spec §6):** at depth 16 *plain*, `mixed`/`split` diverge at every LR while `alternate` + composition-aware init stays stable → `alternate` uniquely trains deep plain stacks. **This is the claim that has NOT been checked on real data.**

## The outstanding experiment — deep *plain* verdict (H-plain)

**Not covered by anything merged or in-flight.** #111 is shallow; #114/#115 are deep *residual*. The flavor-ablation grid runner (`benchmarks/flavor_ablation.py`, merged in #110) is built for exactly this — plain topology, depths {4,8,16}, per-run **divergence-rate** — but **has not been run** (there is no `benchmarks/results/flavor-ablation/` on `main`).

**Run it (GPU, `gpu-torch`):**
```bash
uv sync --group bench
python -m benchmarks.flavor_ablation_launch \
  --datasets heart auto synth_lattice_clow synth_lattice_cmid synth_lattice_chigh \
  --backend torch --devices cuda:0 cuda:1 --out-dir benchmarks/results/flavor-ablation
# LR robustness mini-sweep (depth 8):
python -m benchmarks.flavor_ablation --dataset heart --backend torch --lr-sweep --out-dir benchmarks/results/flavor-ablation
python -m benchmarks.flavor_ablation --dataset auto  --backend torch --lr-sweep --out-dir benchmarks/results/flavor-ablation
```
Then read the verdict off `divergence_rate` vs depth per flavor (does `mixed`/`split` divergence at depth 16 reproduce on real data? does `alternate` stay stable and win there?) and write it into a `docs/benchmarks/` page. **Confirm the runner CLI matches the plan** (`docs/superpowers/plans/2026-07-14-phase4-flavor-ablation.md`); if `flavor_ablation_launch.py`/the depths differ from what #110 actually merged, reconcile first (`grep -n depth benchmarks/flavor_ablation.py`).

## Other open threads

- **`convex_fraction = 0.5` default** — #111 motivates fixing it (drop the searched knob + the off-0.5 init fixed-point). Small `mononet` + config change; own PR.
- **Phase 3 — concepts docs** — distil the three-flavor construction/init/residual writeup (parent spec §3–§6) into `docs/concepts/` (constructions, initialization, residual pages). Pure docs, CPU. Not started.
- **Notebook release-gate** — `docs/benchmarks/*.ipynb` outputs were source-migrated in #108 but not re-executed; run `tools/execute-benchmarks.sh` (GPU) to refresh outputs before tagging `0.0.0a1`.
- **Deferred cleanups** — rename `absolute_init_params`→`mixed_init_params` + keras `_absolute_default`; rename equivalence case filename slugs (`-switch-`/`-abs-`); rename `tests/*/test_absolute_init.py`.

## Pointers

- Parent spec: `docs/superpowers/specs/2026-07-13-monotone-constructions-init-and-ablation-design.md` (three flavors, init math, residual, findings §6).
- Benchmark spec: `docs/superpowers/specs/2026-07-14-flavor-ablation-benchmark-design.md`.
- Plans: `docs/superpowers/plans/2026-07-14-{mode-rename-migration,phase2-alternate-construction,phase4-flavor-ablation}.md`.
- Results: `benchmarks/results/alternate-base/` (#111 shallow); `docs/benchmarks/alternate-base-result.md`.
- Related in-flight: #113 (larger synthetic n_train), #114 (deep-residual accuracy plumbing), #115 (depth-neutral screen).
- Auto-memory: `alternate-construction-init` (in `~/.claude/.../memory/`) mirrors this state.

## What's next (recommended order)

1. **Run the deep-plain grid** (above) — the decisive test of `alternate`'s reason to exist. Cheap-ish focused set; GPU.
2. If deep-plain confirms the CPU prior → write the H-plain verdict doc; if it refutes it (like the depth-neutral residual screen) → document that `alternate` is parity-only and reconsider shipping it as more than an option.
3. **Fix `convex_fraction = 0.5`** default (independent, worthwhile regardless).
4. **Phase 3 concepts docs** (CPU, anytime).
