# Loan size-ladder: does deep monotone residual win with scale?

Status: Approved (design)

Follow-up experiment tracked from PR #72. Branch: `feat/loan-size-ladder`.
Benchmark-only — no `mononet` package change (`benchmarks/` stays out of the wheel).

## 1. Problem / motivation

PR #72's Stage-2 deep-vs-shallow study found a mostly-null result with one
exception: **depth helped only on `loan`** — the largest dataset (~419k train
rows), whose best config was a 14-layer `absolute` residual stack. Everywhere
else (auto-mpg, heart, compas, blog — all smaller) ≤4 effective layers won and
the deep band was neutral-to-worse. That single cross-dataset data point
confounds size with every other way the datasets differ. This experiment turns
it into a controlled dose-response: **hold `loan` fixed, vary the training-set
size N, and measure the deep−shallow gap as a function of N.** If the gap climbs
from ≈0/negative at small N to clearly positive near full size, that is causal
evidence for "depth pays off once the dataset is large enough."

## 2. Goals & constraints

- Reuse the existing `benchmarks/` search + results + protocol harness; add
  only a subsample helper, a run script, a plot/table, a docs page, and a
  RUNBOOK.
- No `mononet` package / kernel / `model_builder` change.
- Respect the standard protocol: the **test set is full and never touched**
  during search; selection is on train-only CV; final numbers are multi-seed on
  the held-out test with the IQM estimator.
- Land plumbing + smoke here (mergeable with no real numbers); the heavy full
  run is the GPU session's job per the RUNBOOK.

## 3. Experiment design

- **Comparison (isolate depth).** Both arms fix `mode="absolute"`,
  `residual=True`; only the depth band differs:
  - **deep** searches `depth ∈ {6, 10, 16}` (L = 14/22/34 layers),
  - **shallow** searches `depth ∈ [1, 4]`.
  This is `suggest_config(..., mode="absolute", residual=True, deep=True|False)`
  in `benchmarks/_common/search_spaces.py`. Holding mode + construction fixed
  makes Δ(N) a clean "does depth pay off" signal, uncontaminated by
  mode/plain-vs-residual choices. (`switch` and other datasets are out of scope;
  a second curve could be added later.)
- **Methodology (per-N tuned — option A).** At each ladder point N, run the full
  standard loan search **independently** for the deep and shallow arms, so each N
  gets its own best-achievable config. This is "best deep vs best shallow *at
  this N*," the fair comparison a scaling claim needs. (Fixed-config or
  re-tune-cheap-HPs-only were considered and rejected: fixing the full-N config
  confounds "depth helps" with "HP tuned for 419k underperforming at small N.")
- **Ladder.** N ∈ `{5_000, 15_000, 45_000, 135_000, full}` (~×3 geometric; the
  top rung is the untouched full train split, anchoring the curve to #72's
  regime). Plotted on a log-N x-axis.
- **Subsampling.** A stratified random subsample of the **train** split to N
  rows, preserving the class ratio, deterministic given a seed. The subsample
  seed is the per-run seed, so the multi-seed test loop varies the subsample
  *and* the initialization together; IQM over seeds absorbs both sources of
  variance. `X_test`/`y_test` are never subsampled.
- **Budget per (N, arm).** The standard loan search: 25 trials, 1-fold CV,
  3 search-seeds; then refit + **10-seed** test on the full test set. (1-fold and
  10 test seeds are loan's existing protocol settings.)
- **Reported per N.** deep IQM, shallow IQM, and **Δ = IQM(deep) − IQM(shallow)**
  with a seed-bootstrap band, plus the selected configs and per-arm collapse
  counts. Metric is accuracy (loan is binary classification).

## 4. Architecture / components

- **`benchmarks/_common/splits.py`** — new `subsample_train(bundle, n, *, seed,
  stratify=True) -> DatasetBundle`: returns a new bundle with `X_train`/`y_train`
  reduced to `n` stratified rows (class ratio preserved), deterministic per
  seed; `X_test`/`y_test` passed through unchanged. If `n >= len(train)`, returns
  the bundle unchanged (the "full" rung). One responsibility, unit-testable.
- **`benchmarks/loan_size_ladder_run.py`** — standalone run script mirroring
  `deep_residual_run.py`. For each `N` in the ladder and each arm in
  `{deep, shallow}`: wrap the loan bundle with `subsample_train`, then reuse the
  existing `_common/search.py` search + `final_eval` primitives with the fixed
  arm config (`mode="absolute"`, `residual=True`, `deep=True|False`). Emits
  `benchmarks/results/size-ladder/loan.json`.
- **Results schema** (`benchmarks/results/size-ladder/loan.json`): a list of
  records `{n, arm, depth, config, test_mean, test_std, test_median, test_iqm,
  collapse}` — same field vocabulary as the Phase-2 result JSONs, plus `n` and
  `arm`.
- **Δ + plot helper** — a small function (in `benchmarks/_common/make_tables.py`
  or a dedicated `size_ladder_report.py`) that reads the JSON, computes Δ(N) with
  a seed-bootstrap band, and renders both a per-N table and a Δ-vs-N plot
  (log-N x-axis) saved as a committed asset.
- **`docs/benchmarks/loan-size-ladder.md`** — the results page: the Δ-vs-N plot
  + per-N table + interpretation. Wired into the benchmarks toctree and
  cross-linked from `docs/benchmarks/deep-residual-accuracy.md`.
- **`benchmarks/RUNBOOK-loan-ladder.md`** — end-to-end run + report procedure for
  the GPU session (mirrors `RUNBOOK-stage2.md`).

## 5. Testing

- **`subsample_train` unit tests** (`tests/benchmarks/`): returned `X_train` has
  exactly `n` rows; class ratio within a small tolerance of the original;
  identical output for the same seed and different for different seeds;
  `X_test`/`y_test` byte-identical to the input; `n >= len(train)` returns the
  bundle unchanged.
- **Run-script smoke test**: end-to-end on a tiny synthetic classification
  bundle (no real loan download) with a 2-point ladder, ≤2 trials, 1 seed —
  asserts the JSON is written with the expected records and that both arms
  produced a finite `test_iqm`. Fast, CI-cheap.
- Existing suites stay green; `uv run mypy` clean across all backends; strict
  `sphinx-build -W` clean (the new docs page + committed plot asset resolve).

## 6. Scope split

- **Landed in this PR (plumbing):** `subsample_train` + tests, the run script +
  smoke test, the Δ/plot helper, the docs page skeleton (with placeholder/smoke
  numbers), the RUNBOOK. Mergeable with no real full-ladder numbers.
- **GPU session (per RUNBOOK):** the real full ladder, especially the top rungs
  (135k, full ~419k); regenerate the plot + table + commit the results JSON and
  final docs page.

## 7. Non-goals / out of scope

- Other datasets; `switch` mode; a second (switch) curve.
- Cross-dataset significance machinery (Friedman/Nemenyi) — as in the protocol.
- Searching `sub_depth` (fixed at 2) — a separate tracked follow-up.
- Any `mononet` package, kernel, or `model_builder` change.

## 8. Open items

- Exact bootstrap band definition (percentile vs normal-approx over the 10 test
  seeds) — pick percentile at implementation; documented on the plot.
- Whether the "full" rung simply reuses #72's existing loan `absolute` deep +
  shallow numbers or is re-run for consistency — RUNBOOK decision; re-running is
  cleaner (same code path for every rung).
