# Deep monotonic residual — real-dataset accuracy (Sub-project B, Stage 2)

**Date:** 2026-07-05
**Status:** Draft (awaiting author review)
**Builds on:**
- Sub-project B Stage 1 ([deep-monotonic-residual](2026-07-03-deep-monotonic-residual-design.md), PR #67) — `MonoResidual.sub_depth` made deep monotone stacks *trainable* on synthetic data.
- The Phase-2a flavor study + standard benchmark protocol ([phase2a-hp-search-and-flavor-study](2026-06-28-phase2a-hp-search-and-flavor-study-design.md), [standard-benchmark-protocol](2026-06-30-standard-benchmark-protocol-design.md)).

## 1. Question & posture

Stage 1 proved *trainability*: with `sub_depth=2` residual skips, monotone stacks train to depth 32 where plain stacks diverge. Stage 2 asks the accuracy question:

> **Does the now-trainable depth improve held-out test accuracy on real datasets, versus the shallow tuned flavors?**

**Honest-result posture.** A null or negative outcome — depth doesn't improve, or mildly hurts, test accuracy on these small/medium tabular datasets — is an **expected and fully reportable finding**, not a failure. (The shallow AutoMPG residual numbers already hint depth may not help on tiny data: residual test MSE 9.94/10.44 vs plain 9.52/9.56 at depth 1–2.) Stage 1 establishes the capability; Stage 2 measures whether the capability translates to accuracy, either way. The paper reports whatever the protocol yields.

## 2. The new flavor: `{mode}-deep`

The study currently has **4 flavors** = `{switch, absolute} × {plain, residual}`, all searching `depth ∈ [1,4]`. Stage 2 adds a **deep** flavor per mode → **6 flavors/dataset**:

| flavor | `residual` | depth search band | effective layers (`sub_depth=2`) |
|---|---|---|---|
| `{mode}-plain` | `False` | `suggest_int(1,4)` | 1–4 |
| `{mode}-residual` | `True` | `suggest_int(1,4)` | ≈ 4–10 |
| `{mode}-deep` | `True` | `suggest_categorical([6,10,16])` | ≈ **14 / 22 / 34** |

Key design fact: **a deep flavor is simply `residual=True` with a larger depth band.** The `BenchmarkConfig` it produces is structurally identical to a `residual` config (both set `residual=True`, both use the default `sub_depth=2`). Therefore:

- **`benchmarks/_common/model_builder.py` is NOT changed.** All three backends already build the residual stack from `cfg.residual`/`cfg.depth`/default `sub_depth=2`. The `depth` block count flows through unchanged. Effective depth = `2·depth + 2` (input `MonoLinear` + `depth` × 2-layer `MonoResidual` `F` + head).
- **`sub_depth` is fixed at 2** (the Stage-1 recommendation) — **not searched**. Searching K is noted as future work.
- The distinction between `residual` and `deep` lives entirely in (a) the **depth search band** and (b) **flavor bookkeeping** (name + result-file suffix).

## 3. Code changes

### 3.1 `benchmarks/_common/search_spaces.py`
`suggest_config(...)` gains `deep: bool = False`. The only behavioural change:
```python
if deep:
    depth = trial.suggest_categorical("depth", [6, 10, 16])
else:
    depth = trial.suggest_int("depth", 1, 4)
```
Everything else (width, lr, weight_decay, dropout, lr_decay, batch_size, convex_fraction, `residual` passthrough) is unchanged, so the deep flavor differs from `residual` by depth band only. (Each flavor is its own Optuna study, so reusing the param name `"depth"` across bands is fine.)

### 3.2 `benchmarks/_common/search.py`
Flavors become **triples** `(mode: str, residual: bool, deep: bool)`.
- `flavor_name(mode, residual, deep=False) -> str`: returns `f"{mode}-deep"` when `deep`, else `f"{mode}-{'residual' if residual else 'plain'}"`.
- `_ALL_FLAVORS` grows to 6:
  ```python
  _ALL_FLAVORS = (
      ("switch", False, False), ("switch", True, False),
      ("absolute", False, False), ("absolute", True, False),
      ("switch", True, True), ("absolute", True, True),
  )
  ```
- `search(...)` and `final_eval(...)` gain `deep: bool = False`; `search`'s objective calls `suggest_config(..., deep=deep)`; both call `flavor_name(mode, residual, deep)`.
- `run_dataset(...)`: the flavor loop unpacks the triple `for mode, residual, deep in flavors:` and threads `deep` into `search`/`final_eval`. The Optuna `study_name`/storage `.db` filename derive from `flavor_name(mode, residual, deep)` so the deep study is separate.
- `_BUDGET` is unchanged: `auto/heart/compas = (50, range(10), 5)`, `loan/blog = (25, range(5), 1)`. **1-fold for loan/blog stays on statistical grounds** (large split → low variance), independent of the now-ample GPU compute.

### 3.3 `benchmarks/search.py` (CLI)
- `_parse_flavors`: accept `kind ∈ {plain, residual, deep}`; map `deep → (mode, True, True)`, `residual → (mode, True, False)`, `plain → (mode, False, False)`.
- Default flavor set = all 6 (`_ALL_FLAVORS`). `--dry-run` echo and `--smoke` preset unchanged in shape (smoke still runs a tiny 2-trial/2-fold subset; it may include a deep flavor to exercise the path).

### 3.4 Result JSON — unchanged schema
Deep results write `results/phase2/<ds>-{switch,absolute}-deep.json` with the **existing** schema (`best_params`, `cv_best`, `test_metric`, `test_mean`, `test_std`, `n_seeds`). No schema change; no migration.

## 4. Reporting & docs

**Page:** a dedicated **`docs/benchmarks/deep-residual-accuracy.md`** (plain Markdown, execution-safe — the docs build runs execution-off), wired into the `docs/benchmarks/index.md` toctree and **cross-linked from `docs/concepts/monotonic-residual.md`** (which owns the trainability story). Placed under `benchmarks/` because it is an accuracy/benchmark artifact alongside `flavor-comparison` and `protocol`.

**Content:** per dataset, a table comparing the **best shallow flavor** (min/max over plain+residual, per the metric's direction) against the **deep flavor**, per mode, reporting `test_mean ± test_std`, the selected depth, and `cv_best`. Plus a short honest read (does trainable depth help / hold / hurt), a reproduce command, and a pointer to `protocol.md`.

**Placeholder until the GPU run:** the page ships with a "results pending the GPU run" note and a static table skeleton (the 5 datasets × modes, cells `—`). The GPU session fills it from the committed `*-deep.json`. Same pattern as Stage 1's Task 7.

## 5. Execution handoff (cross-session)

The heavy search runs on a **different machine + Claude session** in the `gpu-torch` devcontainer (5090 / Blackwell sm_120, merged in #65/#68). This branch merges the **plumbing first** (green CI, no real numbers) so the GPU session starts from clean main.

**This session delivers:** §3 code, per-backend unit tests (§6), the §4 docs skeleton, and **`benchmarks/RUNBOOK-stage2.md`** — the exact handoff:
- `gpu-torch` devcontainer entry + `uv sync` extras.
- The full run: `tools/mononet-benchmark-search --datasets auto,heart,compas,loan,blog` (all 6 flavors by default), with a note that shallow flavors for heart/compas/loan/blog are (re)generated here too since only `auto` exists under the standard protocol, and that the new `sub_depth=2` default makes the prior `auto` residual numbers stale (regenerate all).
- Expected outputs: `results/phase2/<ds>-<flavor>.json` for all 5×6.
- Post-run: commit the JSON, fill the `deep-residual-accuracy.md` table, verify docs build, open the follow-up PR.

**The GPU session owns** the results JSON, the filled docs table, and its own PR.

## 6. Testing / CI

- `test_search_spaces` (or existing): `suggest_config(deep=True)` samples `depth ∈ {6,10,16}`; `deep=False` samples `depth ∈ [1,4]`; all other fields unchanged; `deep=True` implies the caller sets `residual=True`.
- `test_search`: `flavor_name` maps the three kinds correctly; `_parse_flavors("absolute-deep")` → `(("absolute", True, True),)`; `_ALL_FLAVORS` has 6 entries.
- A 2-trial/2-fold synthetic `run_dataset` smoke over one **deep** flavor writes a finite-metric JSON named `*-deep.json` (mirrors the existing smoke test; tiny config, no real search).
- Per-backend: the built model for a deep config has the expected effective depth (a torch structural assertion suffices; jax/keras covered by the existing residual builder tests + the shared `run_dataset` path).
- ruff + `uv run --group bench mypy` + pre-commit + `./tools/build-docs.sh` green. **No real search in CI.**

## 7. Acceptance

- `{mode}-deep` flavor exists for both modes; `model_builder` unchanged; `sub_depth=2` fixed.
- Deep depth band `{6,10,16}`; shallow bands unchanged; result JSON schema unchanged.
- Unit/smoke tests + docs build green; plumbing PR mergeable with no real numbers.
- `RUNBOOK-stage2.md` gives the GPU session an unambiguous end-to-end run + report procedure.
- Docs page skeleton wired into the benchmarks toctree and cross-linked from `monotonic-residual.md`.

## 8. Non-goals / out of scope

- Running the real search here (GPU session's job).
- Searching `sub_depth` (fixed at 2; future work).
- Changing `mononet` package code, kernels, or `model_builder`; `benchmarks/` stays out of the wheel.
- Cross-dataset significance machinery (Friedman/Nemenyi) — deferred, as in the protocol spec.
- Normalization layers or new composed model classes.

## 9. Open items

- **Deep band values.** `{6,10,16}` blocks (≈14/22/34 effective layers) is the approved default; the GPU session may widen to include a larger point if results warrant, but that is a post-hoc extension, not part of this plumbing.
