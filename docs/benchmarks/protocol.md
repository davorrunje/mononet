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
