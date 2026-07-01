# Benchmark protocol

`mononet`'s benchmarks use the standard held-out protocol for comparing tabular
models. For each dataset:

1. **Fixed splits.** We use the published `train_<ds>.csv` / `test_<ds>.csv`
   (Zenodo 10.5281/zenodo.7968969). The test set is touched exactly once, for the
   final report — never for any model-selection decision.
2. **Model selection on cross-validation only.** Hyperparameters and epochs are
   chosen on a *k*-fold cross-validation of the **train** split (stratified for
   classification). Folds: 5 for the small/medium datasets (Auto MPG, Heart,
   COMPAS); a single 80/20 holdout for the large ones (Loan, Blog), where a single
   split is already low-variance and *k*-fold would cost 5× for no real gain.
   The per-trial objective is the **mean metric across folds**.
3. **Refit + multi-seed test.** The single selected configuration is refit on the
   full train split and evaluated on the held-out test set across **10 seeds**
   (parameterisable).
4. **Reporting.** We report the **mean ± standard deviation over all seeds**. We do
   **not** select a best-*k* subset of seeds.

## Why our numbers differ from the original papers

The numbers quoted in Runje & Shankaranarayana (2023) and the prior baselines they
compared against were produced by a different protocol — inherited, via the
[`airtai/monotonic-nn`](https://github.com/airtai/monotonic-nn) reference code, from
those earlier papers. In that protocol the **test set is used as the validation
set**: hyperparameters are tuned with `validation_data=test`, early stopping
monitors the test loss, the per-run score is the **best epoch on the test curve**,
and the reported figure is the **mean of the best 5 of 10 runs**.

That makes those numbers optimistic by construction — the test set drives model
selection. Our protocol never lets the test set influence any choice, so our
held-out results sit somewhat **higher (worse)** than the published figures. The
difference is expected and is **not** a regression in `mononet`; the two sets of
numbers are simply **not directly comparable**. We keep the published figures in the
comparison tables for reference, labelled `[prior protocol]`.

## Interpreting the numbers

Two things to keep in mind when reading the tables, both illustrated by a diagnostic
run on Auto MPG (the smallest dataset, 314 train / 78 test):

**The CV-selection score is not a test estimate.** The CV metric that drives
hyperparameter selection is systematically *optimistic* relative to held-out test
error — it is the minimum over many trials, so it partly selects luck. On Auto MPG,
nested cross-validation (which re-runs the whole search inside each outer fold) puts
the honest pipeline estimate roughly **midway** between the CV-selection score and the
published-split test score: of the ~2 MSE gap, about half is this selection optimism
and about half is the published 78-row test split being genuinely harder than
train-distribution holdouts. Report and compare the **test** column, never the CV one.

**Small datasets are noisy — don't over-read single-dataset margins.** On Auto MPG the
per-fold spread in nested CV is large (±1–4.5 MSE, with occasional divergent folds),
so flavor differences smaller than ~1 MSE are within the noise. Treat per-dataset
flavor rankings as suggestive; a robust "which flavor wins" conclusion needs the
larger datasets (COMPAS ≈ 5k, Blog ≈ 47k, Loan ≈ 419k rows), where these estimates
tighten considerably.
