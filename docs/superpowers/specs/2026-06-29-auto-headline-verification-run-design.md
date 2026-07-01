# AutoMPG Full-Budget Verification Run — Design

**Date:** 2026-06-29
**Status:** Superseded (reporting protocol)
**Builds on:** [Phase 2a search execution](2026-06-29-phase2a-search-execution-design.md) (PR #57, merged) — `run_dataset()`, the `python -m benchmarks.search` CLI, `flavor-comparison.ipynb`.

> **Superseded by [Standard Benchmark Protocol](2026-06-30-standard-benchmark-protocol-design.md).**
> This doc's reporting protocol (single holdout, "10 final seeds (best-5)", `val_best`)
> reflects the original inherited scheme; the AutoMPG numbers are now produced under the
> standard held-out protocol (k-fold CV for HP selection, mean±std over **all** seeds,
> `cv_best`). The goals/health-check rationale below still stand.

> Execution/validation task, not new modelling. Uses the merged Phase-2a tooling
> as-is. No `mononet` package change; no new benchmark code.

## 1. Goals

1. **Pipeline health.** Prove the full Phase-2a search → final-eval → JSON →
   rendered-table path works end-to-end on *real* Zenodo data at full committed
   budget on this CPU MacBook (M4 Pro, torch backend, CPU device).
2. **Hypothesis check.** Test the prior assumption that **residual variants
   outperform plain** on AutoMPG, comparing within each mode.

## 2. What runs

1. `python -m benchmarks.datasets.download --dataset auto` → caches the Zenodo
   CSV under `~/.cache/mononet/datasets` (never committed).
2. `tools/mononet-benchmark-search --datasets auto --backend torch --n-jobs N`
   — all 4 flavors (`switch/absolute × plain/residual`), full committed budget
   from `_BUDGET["auto"]` = 50 trials, 10 final seeds (best-5), epochs 50.
   Flavors run sequentially; `--n-jobs` threads trials within each study to use
   the many cores (partial speedup — torch releases the GIL). Run in the
   background. Writes `benchmarks/results/phase2/auto-<flavor>.json` ×4.

## 3. Success criteria (pipeline health)

- All 4 JSON written; every `val_best` / `test_mean` / `test_std` finite (no
  NaN/inf).
- Test MSE in a sane band — paper AutoMPG ≈ **8.37**, so single-to-low-double-digit
  MSE is expected. Wildly off ⇒ a plumbing bug, not a result.
- `flavor-comparison.ipynb` renders the populated auto row (flavors vs
  paper-quoted vs XGBoost) cleanly under the docs build.

## 4. Hypothesis check (reported, not a gate)

Compare `test_mean` (MSE, lower = better) within each mode:
- `switch-plain` vs `switch-residual`
- `absolute-plain` vs `absolute-residual`

Report what the data shows, **including a result that contradicts the
residual-better assumption**. One dataset × 10 seeds is a weak signal — framed
as indicative, not conclusive. Use `test_std` to judge whether any gap is
within noise.

## 5. On success → commit as real numbers

If the run is clean, commit on this branch (never main):
- the 4 `benchmarks/results/phase2/auto-*.json`
- the re-rendered `docs/benchmarks/flavor-comparison.ipynb`

→ signed commits → PR. The other 4 datasets remain for the maintainer's full
`gpu-torch` run. Do **not** commit `*.db` / `*.jsonl` study artifacts.

## 6. Non-goals

- The other 4 datasets (maintainer's full run).
- Any code change to `run_dataset`/CLI/`mononet`.
- A conclusive flavor verdict — this is a single-dataset health + signal check.
