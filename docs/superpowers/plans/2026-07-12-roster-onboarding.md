# Roster onboarding + multi-dataset screen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Onboard four monotone binary-classification datasets (Taiwan Credit, Polish Bankruptcy, German Credit, Lending Club-Zenodo) via the proven Adult recipe, and add a multi-dataset screen launcher so the max-size screen runs several datasets concurrently across both GPUs.

**Architecture:** Each dataset = a `DataSource` (`sources.py`) + `DatasetSpec` (`spec.py`) + a `prepare/<name>.py` (raw → mononet CSV, tested on a synthetic sample) + committed LFS `.csv.gz` + a `_BUDGET` entry. The launcher distributes datasets across GPUs, one tested `screen_dataset` process each. Follows Phase-1's Adult task (`benchmarks/datasets/prepare/adult.py`) as the template.

**Tech Stack:** Python 3.11+, pandas, scipy/liac-arff (Polish ARFF), sklearn, matplotlib, Optuna, PyTorch, Git LFS, uv.

## Global Constraints

- Python 3.11+, ruff line length 88; strict mypy; MyST field-list docstrings (no `:type:`/`:rtype:`); stdlib dataclasses (no Pydantic).
- Benchmark-only: NO `mononet/` change; `benchmarks/` out of the wheel.
- Loader does `float(v)` on every cell → prep output must be all-numeric (one-hot every non-monotone categorical; monotone ordinals kept raw).
- All datasets binary classification (target `ground_truth`, 0/1).
- Hosting: all four are redistributable (UCI CC-BY / Zenodo CC-BY) → committed to Git LFS under `benchmarks/data/<name>/`.
- Test split never mixed with train. Prep deterministic (fixed seed / documented chronological rule).
- Monotone columns must be preserved (not one-hot'd) and their names must match the `DatasetSpec` exactly. SEX/RACE/age-like protected or contested columns are NOT constrained.

## Shared recipe (every dataset task)

Mirror `benchmarks/datasets/prepare/adult.py`:
1. `prepare_<name>(raw: pd.DataFrame) -> tuple[train, test]` — recode target to `ground_truth` (0/1); one-hot all non-monotone categoricals on the FULL frame before splitting (column alignment); keep monotone ordinals raw & numeric; return numeric train/test.
2. Add `SOURCES["<name>"] = DataSource(...)` (hosting `lfs`, real URL + license).
3. Add the `DatasetSpec` to `DATASETS_SPEC`.
4. Add a `_BUDGET` entry: `(25, range(10), 5)` for the small adult-scale datasets (taiwan/polish/german); `(25, range(10), 1)` for the large LC (like loan/blog). Only used by the future standard-roster path; the screen uses run_ladder defaults.
5. TDD test on a SYNTHETIC raw sample (no network): asserts `ground_truth` 0/1, all-numeric output, monotone columns present, split sizes/no train/test overlap, class ratio preserved.
6. Materialize the real data (download → prep → `.csv.gz`) and commit via LFS; `git lfs ls-files | grep <name>` must show pointers.

---

### Task 1: Taiwan Credit (`taiwan`)

**Facts:**
- Source: UCI 350. Download `https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip` → `.xls`; read with `pandas.read_excel(path, header=1)` (two-row header — `header=0` corrupts names).
- Target: `default payment next month` → `ground_truth` (already 0/1).
- Monotone-increasing: `PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6` (delinquency ↑ → default ↑).
- Monotone-decreasing: `LIMIT_BAL` (↑ → default ↓), `PAY_AMT1..PAY_AMT6` (↑ → default ↓).
- Do NOT constrain: `SEX, EDUCATION, MARRIAGE, AGE`, `BILL_AMT1..6` (utilization-confounded). Drop `ID`.
- All columns already numeric → minimal/no one-hot; split stratified 80/20, seed 0.

- [ ] Step 1: synthetic-sample prep test (RED) → Step 2: implement `prepare/taiwan.py` + `MONO_INCREASING`/`MONO_DECREASING` → Step 3: GREEN.
- [ ] Step 4: register `DataSource` + `DatasetSpec` (`("PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")` increasing; `("LIMIT_BAL","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6")` decreasing) + `_BUDGET`.
- [ ] Step 5: materialize + LFS-commit `benchmarks/data/taiwan/{train,test}_taiwan.csv.gz`; verify loader loads + `git lfs ls-files`.
- [ ] Step 6: mypy clean; commit.

---

### Task 2: Polish Bankruptcy (`polish`)

**Facts:**
- Source: UCI 365. Download `https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip` → use `3year.arff` (3-yr horizon, common default). Parse ARFF with `scipy.io.arff.loadarff` (stdlib-adjacent via scipy, already a dep) → DataFrame; decode byte target.
- Target: `class` → `ground_truth` (0/1). Attr* have missing values (`?`) — drop rows with NA in the monotone columns (document it) or median-impute; prefer dropna on the kept columns for determinism.
- Monotone-increasing: `Attr2` (liabilities/assets, leverage ↑ → bankruptcy ↑).
- Monotone-decreasing: `Attr1` (ROA), `Attr4` (current ratio), `Attr17` (assets/liabilities), `Attr23` (net margin), `Attr35` (profit-on-sales/assets) — all ↑ → bankruptcy ↓.
- Keep ALL 64 `Attr*` features (all numeric); median-impute missing (`?`/NaN) values per column for determinism; constrain only the 6 monotone Attrs (mirrors `loan`: monotone constraints amid free features). No categoricals.
- Split stratified 80/20, seed 0. Prep test asserts all 64 feature columns retained + the 6 monotone columns present + no NaN in output.

- [ ] Step 1: synthetic-sample prep test (RED) → Step 2: implement `prepare/polish.py` → Step 3: GREEN.
- [ ] Step 4: register `DataSource` + `DatasetSpec` (`("Attr2",)` increasing; `("Attr1","Attr4","Attr17","Attr23","Attr35")` decreasing) + `_BUDGET`.
- [ ] Step 5: materialize + LFS-commit; verify.
- [ ] Step 6: mypy clean; commit.

---

### Task 3: German Credit (`german`)

**Facts:**
- Source: UCI 144. Download `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data` (space-separated, no header, coded categoricals). Assign the 20 documented column names + target.
- Target: last column, `1=good, 2=bad` → recode `ground_truth = (target == 2).astype(int)`.
- Monotone-increasing: `duration` (attr2), `credit_amount` (attr5), `installment_rate` (attr8).
- Monotone-decreasing: `age` (attr13).
- One-hot all coded categorical attributes (checking status, credit history, purpose, savings, employment, personal-status/sex, debtors, property, other-plans, housing, job, telephone, foreign-worker); keep the 4 monotone numerics raw. Numeric non-monotone (`residence_since`, `existing_credits`, `people_liable`) kept raw but unconstrained.
- Split stratified 80/20, seed 0.

- [ ] Step 1: synthetic-sample prep test (RED) → Step 2: implement `prepare/german.py` (column names + recode + one-hot) → Step 3: GREEN.
- [ ] Step 4: register `DataSource` + `DatasetSpec` (`("duration","credit_amount","installment_rate")` increasing; `("age",)` decreasing) + `_BUDGET`.
- [ ] Step 5: materialize + LFS-commit; verify.
- [ ] Step 6: mypy clean; commit.

---

### Task 4: Lending Club-Zenodo (`lc`)

**Facts:**
- Source: Zenodo 11295916, CC-BY. Download `https://zenodo.org/records/11295916/files/LC_loans_granting_model_dataset.csv?download=1` (~160 MiB, 1.35M rows).
- Keep the 8 features: `revenue, dti_n, loan_amnt, fico_n, experience_c, emp_length, purpose, home_ownership_n`; target `Default` → `ground_truth` (0/1).
- Monotone-increasing: `dti_n`. Monotone-decreasing: `fico_n`, `revenue`. (The 3 the source paper constrains; `loan_amnt` left unconstrained per the paper's Table A.1.)
- One-hot the categoricals `emp_length, purpose, home_ownership_n`; `experience_c` binary→int; keep `dti_n, fico_n, revenue, loan_amnt` raw numeric.
- **Chronological split on `issue_d`** (NOT stratified/random): train = issued 2007–2015 (~829,347), **exclude 2016 entirely**, test = 2017–2018 (~225,277). Parse `issue_d` (e.g. "Dec-2015") to a year; document the rule.
- Store gzipped (`.csv.gz`) — large; confirm LFS pointer + note the on-disk gz size.

- [ ] Step 1: synthetic-sample prep test (RED) — build a tiny synthetic frame with `issue_d` years spanning 2014–2018 and assert the chronological split (2016 excluded; train years ≤2015; test years ≥2017), `ground_truth` 0/1, monotone cols present, all-numeric. → Step 2: implement `prepare/lc.py` → Step 3: GREEN.
- [ ] Step 4: register `DataSource` + `DatasetSpec` (`("dti_n",)` increasing; `("fico_n","revenue")` decreasing) + `_BUDGET`.
- [ ] Step 5: materialize (download 160 MiB → prep → gz) + LFS-commit `benchmarks/data/lc/{train,test}_lc.csv.gz`; verify loader loads (~829k train rows) + `git lfs ls-files`.
- [ ] Step 6: mypy clean; commit.

---

### Task 5: Multi-dataset screen launcher

**Files:** Create `benchmarks/screen_launch.py`; Test `tests/benchmarks/test_screen_launch.py`.

**Interfaces:**
- Consumes: `screen_dataset` (from `benchmarks.large_screen_run`), `merge`-style JSON handling.
- Produces: `merge_screens(paths) -> list[dict]` (pure, unit-testable: read per-dataset screen records, sort by `name`); and `main()` CLI (`--datasets a,b,c`, `--devices cuda:0,cuda:1,...`, budget flags, `--out-dir`) that runs one `screen_dataset` **subprocess per dataset** pinned across the device pool (round-robin, like `loan_ladder_launch`), each writing `results/screen/<name>.json`, then merges into `results/screen/all.json`.

- [ ] Step 1: unit test `merge_screens` on two fake per-dataset JSONs → concatenated, sorted by name (RED).
- [ ] Step 2: implement `screen_launch.py` — reuse the device-queue/round-robin pattern from `benchmarks/loan_ladder_launch.py` (one process per dataset via `python -m benchmarks.large_screen_run --dataset <d> --out results/screen/<d>.json`, `$MONONET_TORCH_DEVICE` pinned); `merge_screens` assembles `all.json`. → Step 3: GREEN (merge test).
- [ ] Step 4: mypy clean; commit.

---

## After the tasks (controller, not a subagent task)

Run the concurrent screen on the GPUs per `RUNBOOK-large-screen.md`:
`python -m benchmarks.screen_launch --datasets adult,taiwan,polish,german,lc --devices cuda:0,cuda:1 ...`
(datasets distributed across both GPUs → real multi-dataset utilization). Then render the screen table + plot (`screen_report`), fill `docs/benchmarks/large-dataset-screen.md` with the per-dataset Δ + verdicts, commit results + docs, PR.

## Self-review notes

- Coverage: each dataset → its own task with source URL, monotone tuples (exact), split rule, and prep-test contract; the launcher → Task 5. The concurrent screen run is a controller step (needs GPUs + committed data), not a unit task.
- Type consistency: every `prepare_<name>` returns `(train, test)` DataFrames; every `MONO_*` tuple matches its `DatasetSpec`; `screen_dataset` (unchanged) already returns the 9-key record `screen_launch` merges.
- Risk: LC-Zenodo is a 160 MiB download + large gz commit (Task 4 heaviest); Polish restricts to 6 monotone Attrs for a clean feature set (documented simplification).
