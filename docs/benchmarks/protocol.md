# Benchmark protocol

`mononet`'s benchmarks use the standard held-out protocol for comparing tabular
models. For each dataset:

1. **Fixed splits.** We use the published `train_<ds>.csv` / `test_<ds>.csv`
   (Zenodo 10.5281/zenodo.7968969). The test set is touched exactly once, for the
   final report — never for any model-selection decision.
2. **Stability-aware model selection on cross-validation only.** Hyperparameters
   and epochs are chosen on a *k*-fold cross-validation of the **train** split
   (stratified for classification). Folds: 5 for the small/medium datasets (Auto
   MPG, Heart, COMPAS); a single 80/20 holdout for the large ones (Loan, Blog),
   where a single split is already low-variance and *k*-fold would cost 5× for no
   real gain. Each trial is evaluated over **`search_seeds` (default 3) seeds per
   fold**, and the per-trial objective is the **risk-adjusted one-sigma bound**
   — `mean − std` for maximize metrics, `mean + std` for minimize metrics — over
   all fold×seed runs, **not** the plain fold mean. Multiple seeds per fold expose
   seed-dependent training collapses that a single-seed CV misses, and the
   variance penalty steers the search away from fragile HP regions that train well
   on average but collapse on some seeds (see the collapse note below).
3. **Refit + multi-seed test.** The single selected configuration is refit on the
   full train split and evaluated on the held-out test set across **20 seeds** for
   the small/medium datasets (**10** for Loan/Blog, whose single-holdout final_eval
   is already near-deterministic); parameterisable.
4. **Multi-protocol reporting.** For each (dataset, flavor) we report several
   estimators over the seed runs, so the numbers can be read both
   paper-comparably and robustly:
   - **mean ± std** — the paper-comparable protocol.
   - **median** and **IQM** (interquartile mean: the mean of the middle 50% after
     trimming the top and bottom 25%, following Agarwal et al., *Deep RL at the
     Edge of the Statistical Precipice*, NeurIPS 2021) — robust to the occasional
     seed collapse *and* to lucky runs.
   - **collapse count** — number of seeds that degenerated (a constant
     base-rate-prediction for classification; a gross bad-side outlier for
     regression), reported as an explicit stability metric rather than hidden.

   We do **not** drop only the worst seeds (one-sided trimming biases results
   optimistically); the robust estimators trim symmetrically or by rank.

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

## Training-stability findings behind the protocol

Three concrete findings shaped the model and the selection/reporting choices
above. They are recorded here because they matter for the paper's methodology.

**1. The read-out head must be linear (`identity`), not a nonlinear activation.**
Every monotone layer applies an activation; if the final 1-unit read-out also
does, a `mixed`-mode head becomes `relu(|W|·h + b) ≥ 0`, forcing the
pre-sigmoid non-negative → the model predicts the positive class for every row →
binary classification **collapses to the base rate** (and ReLU's dead zone locks
it there). `split` is spared (its head is a *difference* of activations) and
regression is spared (positive targets), which masks the issue. The fix is a
**linear monotone read-out** — `|W|·h + b` via a first-class `identity` activation
— which is the correct output-layer form and preserves monotonicity. A regression
test asserts the head is linear and that mixed-mode classification clears the
base rate.

**2. Plain monotone stacks explode with depth; use residual for depth.** A plain
`|W|`/split stack is variance-amplifying: on standard-normal input the
activation std grows ~4–5× per layer (≈10¹² over 20 layers), because `|W|` is a
non-negative matrix whose Perron growth compounds the convex/concave mean
structure. **No scalar init fixes it** (shrinking the init only lowers the growth
factor, never below 1). `MonoResidual` keeps the stack conditioned (std ≈ 1.4 over
20 blocks) and is monotone by construction — so the *deep* flavor is the residual
construction searched at larger depths (6/10/16), not a deep plain stack. The
paper's original networks avoided the problem by staying shallow (2 layers).

**3. Aggressive HP regions collapse on some seeds — hence stability-aware
selection + robust reporting.** With a single-seed CV, the search can select a
fragile configuration (e.g. high learning rate × high weight decay on a plain
split stack) that trains well on the CV seed but drives the weights to a dead
attractor (`σ(W⁺x+b) − σ(W⁻x+b) → 0`, constant 0.5, loss `ln 2`) on a fraction of
final-eval seeds. This is not fixable by a single knob — the collapse is
non-monotone in weight decay (both high and zero WD collapse; gradient clipping
makes it worse) — so it is handled two ways: the **stability-aware objective**
(step 2) penalises the variance and avoids the fragile region, and the
**multi-protocol reporting** (step 4) surfaces any residual instability via the
median/IQM and the collapse count instead of hiding it in a mean.
