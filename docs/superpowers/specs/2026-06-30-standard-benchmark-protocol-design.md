# Standard Benchmark Protocol — Design

**Date:** 2026-06-30
**Status:** Draft (awaiting author review)
**Builds on:** [Phase 2a search execution](2026-06-29-phase2a-search-execution-design.md) (PR #57, merged) and the open AutoMPG verification run (PR #62). Supersedes the inherited reporting protocol.

> We discovered the paper-quoted numbers (e.g. AutoMPG MSE 8.37) were produced by a
> protocol — inherited from the prior comparison papers via `airtai/monotonic-nn` —
> that selects on the **test** set: hyperparameters tuned with `validation_data=test`,
> early stopping on test loss, per-epoch best-test-MSE, then **best-5-of-10** runs. That
> is optimistic by construction. mononet adopts a standard held-out protocol instead.
> Our numbers will be **higher (worse) but honest and reproducible**, and not directly
> comparable to the prior papers — this divergence is documented, not hidden.

## 1. Goals & non-goals

### Goals
- Replace the inherited reporting rule with the **modern standard tabular-DL protocol**:
  fixed published train/test split; **k-fold CV on train** for all model selection;
  refit the selected config on full train; **report mean ± std over all seeds** on the
  test set (no best-k).
- **Document the methodology and why our numbers differ** from the prior papers, in the
  benchmark docs (a dedicated protocol section + reframed paper-quoted column + fixed
  Phase-1 "calibration anchor" wording).
- Re-run **AutoMPG** under the new protocol so PR #62 ships honest numbers from the start.

### Non-goals (deferred)
- Cross-dataset significance machinery (Friedman + Nemenyi / critical-difference). Useful
  once all 5 datasets are run; out of scope here.
- Re-running the other 4 datasets (maintainer's full run).
- The harness **architecture** divergence (two-branch embed vs the reference's unified
  mixed-indicator `MonoDense`) — tracked separately; this spec changes *protocol*, not the
  model.
- No `mononet` package change; `benchmarks/` stays out of the wheel.

## 2. The protocol

Per Gorishniy et al. 2021 (*Revisiting DL for Tabular Data*), Grinsztajn et al. 2022, and
Demšar 2006 (cross-model statistics, deferred):

1. **Fixed splits.** Keep the published `train_<ds>.csv` / `test_<ds>.csv`. Test is touched
   exactly once, for the final report — never for any decision.
2. **Model selection on CV only.** Every choice (HPs, epochs) is made on a CV of the train
   split, controlled by a parameter **`n_splits` (default 5)**. `n_splits ≥ 2` ⇒ (stratified
   for `binary_classification`, plain for `regression`) `KFold`; **`n_splits = 1` ⇒ a single
   80/20 holdout** (for large datasets where CV cost isn't worth it). Deterministic given a
   seed. `n_splits` is overridable globally (CLI `--cv-folds`) and has a **per-dataset
   default** so we can pick e.g. 5 for small datasets and 3 or 1 for the large ones.
3. **Per-trial objective = mean CV metric.** Each Optuna trial trains on the fold's train
   indices, evaluates on its val indices, and the objective is the **mean metric across the
   `n_splits` folds** (a single value when `n_splits = 1`).
4. **Final eval.** Refit the single selected config on the **full train** split; run a
   parameterised number of **seeds (default 10)**; evaluate each on the **fixed test set**;
   report **mean ± std over all seeds**. No best-k selection.

This keeps what our pipeline already does right (HPs never see test) and removes the only
non-standard piece (best-5-of-10), while making HP selection robust on small datasets via
CV instead of a single noisy ~60-row holdout.

## 3. Code changes

`benchmarks/_common/splits.py` — add:
```python
def cv_splits(
    bundle: DatasetBundle, *, n_splits: int = 5, seed: int, stratify: bool | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(train_idx, val_idx), ...] over bundle.X_train/y_train.

    n_splits >= 2 -> (stratified for binary_classification) KFold, one entry per fold.
    n_splits == 1 -> a single 80/20 holdout, returned as a one-element list (reuses
    train_val_split's logic). Deterministic given seed; never touches X_test/y_test.
    """
```
Keep `train_val_split` (the `n_splits == 1` path delegates to it).

`benchmarks/_common/search.py`:
- `search(...)` gains `n_splits: int = 5`. The per-trial objective averages the metric over
  the `cv_splits` folds (each fold reuses the existing `run()` via the val-bundle trick: put
  the fold's val rows into the bundle's `X_test/y_test` slot with `dataclasses.replace`).
  `StudyResult.best_value` becomes the **mean CV metric** of the best trial.
- `final_eval(...)`: report **all seeds** — drop `top_k` selection (report mean ± std over
  the full `seeds`). The `Aggregate` keeps `mean`/`std`/`n_seeds`; `n_selected == n_seeds`.
- `run_dataset(...)`: drop `final_top_k`; thread `n_splits` (param, default from `_BUDGET`).
  The `_BUDGET` table carries per-dataset `(n_trials, seeds, n_splits)` — `n_splits` is
  **5 for auto/heart/compas** (tiny/medium: ~62/48/987-row holdout, CV cuts real variance)
  and **1 for loan/blog** (large: ~84k/9.5k-row holdout is already low-variance, so a single
  holdout avoids a pointless 5× search cost). All overridable via `--cv-folds`.

`benchmarks/search.py` (CLI): drop `--final-top-k`; add `--cv-folds` (default: per-dataset
`_BUDGET`, overridable). `--smoke` preset uses `--cv-folds 2` for speed. Seeds stay the
existing `--final-seeds` parameter (default 10).

## 4. Results JSON

Each `results/phase2/<ds>-<flavor>.json` records: `best_params`, `cv_best` (mean CV metric),
`test_metric`, `test_mean`, `test_std`, `n_seeds`. Remove `n_selected` (now == `n_seeds`).
Existing four `auto-*.json` are regenerated under the new protocol.

## 5. Docs (the explanation)

- **New `docs/benchmarks/protocol.md`** (wired into the benchmarks toctree): states the
  4-step protocol above and a "Why our numbers differ from the original papers" section —
  the prior protocol selected on test (HP search, early stopping, per-epoch, and best-5-of-10
  all used the test set), inflating results; mononet reports a standard held-out CV result
  with all-seeds mean±std, so our numbers sit somewhat higher and are **not directly
  comparable**, but are honest and reproducible.
- **`docs/benchmarks/flavor-comparison.ipynb`**: add a short protocol note; relabel the
  paper row `paper (CMNN) [prior protocol — see protocol.md]` so the comparison isn't read
  as apples-to-apples.
- **Phase-1 `auto-mpg.ipynb` / `tables.ipynb`**: replace the "8.37 … calibration anchor for
  checking the harness is wired correctly" wording — a correctly-wired *clean* harness should
  **not** reproduce 8.37; point to `protocol.md`.
- **`benchmarks/README.md`**: add a "Benchmark protocol" subsection mirroring `protocol.md`.

## 6. Testing / CI

- `test_splits.py`: `cv_splits` returns `n_splits` folds whose val indices partition the
  train rows exactly once (for `n_splits ≥ 2`); `n_splits == 1` returns a single ~80/20
  holdout; stratification preserves class balance for binary tasks; determinism for a fixed
  seed; test set never referenced.
- `test_search.py`: a 2-trial, 2-fold synthetic `search()` returns a finite `cv_best`; the
  per-trial objective is the fold-mean.
- `test_run_dataset` / CLI tests: updated for the dropped `top_k` and added `--cv-folds`;
  result JSON has `cv_best`/no `n_selected`.
- ruff + mypy + pre-commit + static-analysis clean; docs build green; no real search in CI.

## 7. Re-run & PR

Re-run AutoMPG (all 4 flavors) under the new protocol on this CPU box; commit the regenerated
`auto-*.json` + re-rendered `flavor-comparison.ipynb`. Fold into PR #62, updating its title/
body to "Adopt standard benchmark protocol + honest AutoMPG numbers (+ search-wrapper fix)".

## 8. Acceptance

- HP selection uses 5-fold CV; reporting is mean±std over all seeds (no best-k).
- `protocol.md` exists and explains the divergence from prior papers; flavor-comparison and
  Phase-1 wording reframed; README updated.
- AutoMPG regenerated under the new protocol; numbers documented as honest/not-comparable.
- Tests/CI green; no package or wheel change.

## 9. Open items

- **Seed count.** Parameterised, default 10 (standard 10–15); bump later if std is noisy.

Resolved: per-dataset `n_splits` defaults are auto/heart/compas = 5, loan/blog = 1 (see §3).
