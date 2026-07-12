# Large-dataset monotonic-depth screen (tabular)

Status: Approved (design)

Phase 1 of the **large-dataset monotonic-depth benchmark** program
([program note](2026-07-11-large-dataset-benchmark-program.md)); an expansion of
Sub-project C (extended benchmarks). Benchmark-only — no `mononet` package,
kernel, or `model_builder` change (`benchmarks/` stays out of the wheel).

## 1. Problem / motivation

PR #72's deep-vs-shallow study and the follow-up loan size-ladder
([design](2026-07-10-loan-size-ladder-design.md)) found depth's payoff is, at
best, confined to the single largest dataset (`loan`, ~419k). That is one data
point. To learn whether "monotone depth pays off at scale" is real, we need
**more large, genuinely monotonic tabular datasets** and a cheap, uniform test
of depth on each. This sub-project assembles that roster, runs a **max-size
deep/shallow screen** per dataset, and routes each dataset by a fixed gate to
either a full size-ladder study (when depth clearly helps at max size) or into
the standard benchmark (when it does not). Data is version-controlled (Git LFS
now, Zenodo later) where licensing permits.

## 2. Goals & constraints

- Reuse the existing `benchmarks/` search + results + protocol harness and the
  size-ladder Δ/bootstrap machinery. Add datasets, a data-hosting layer, a gate,
  a report, docs, and a RUNBOOK.
- Respect the standard protocol: the **test set is full and never touched**
  during search; selection is train-only CV; final numbers are multi-seed on the
  held-out test with the IQM estimator.
- **Tabular binary-classification / regression only.** Learning-to-rank and
  curve-regression are separate program phases (see the program note).
- Redistribution-safe data handling: LFS/Zenodo only for datasets whose license
  permits re-hosting; a preprocessing script for the rest.
- No `mononet` package / kernel / `model_builder` change.

## 3. Dataset roster

All binary classification. Each gets a `DatasetSpec` with domain-justified
monotone-feature lists. `Rows` is the train split (order of magnitude).

| Dataset | Rows | Source / License | Hosting | Clean monotone features |
|---|---|---|---|---|
| ACSIncome (Folktables) | ~1.66M | US Census PUMS, public domain | LFS | `WKHP`↑ (hours), `SCHL`↑ (education); `AGEP`↑ plausible |
| Lending Club (Zenodo, Ariza-Garzón 2024) | ~829k | Zenodo, CC-BY | LFS | DTI↑ → risk↑; grade/score↓ → risk↑ |
| Home Credit Default Risk | ~307k | Kaggle (ToS) | script | ext. score↓ → risk↑; DTI↑ → risk↑ |
| Give Me Some Credit | ~150k | OpenML | LFS if license permits, else script | delinquencies↑ → risk↑; age↓ → risk↑ |
| Adult / Census Income | ~48k | UCI, CC-BY | LFS | education↑, hours↑, capital-gain↑ → income↑ |
| Taiwan Credit (default of CC clients) | ~30k | UCI, CC-BY | LFS | `PAY_*` delinquency↑ → default↑ |
| Polish Bankruptcy (year 3) | ~10k | UCI, CC-BY | LFS | leverage↑ → bankruptcy↑; liquidity↓ → ↑ |
| German Credit (Statlog) | ~1k | UCI, CC-BY | LFS | small; breadth only |

Notes:

- **ACSIncome is the large anchor** (1.66M, public-domain, one-line `folktables`
  fetch) and pairs with Adult (same income target, 48k) for a within-domain size
  contrast. ACSIncome is categorical-heavy (`OCCP`, `POBP`, … are nominal);
  constrain **only** the ordinal/continuous monotone subset, and deliberately do
  **not** constrain `SEX`/`RAC1P` (fairness).
- The existing `loan` (Lending Club, ~419k, Zenodo 7968969, 28 features / 5
  declared monotone) is kept and screened as-is; Lending Club (Zenodo 2024) is
  its **larger sibling** (829k, 8 features, chronological split), not independent
  evidence — the report flags the shared provenance.
- Fannie/Freddie mortgage (tens of millions, textbook monotone credit) is a
  **backlog** item: license forbids re-hosting → manual-download README +
  script-only, deferred (see program note).

## 4. Architecture / components

### 4.1 Data plumbing (`benchmarks/datasets/`)

- **Registry.** One `DatasetSpec` per new dataset (target, `mono_increasing`,
  `mono_decreasing`). No change to `DatasetSpec` itself.
- **Source descriptor.** A per-dataset record of hosting class — `lfs` (path
  under `benchmarks/data/<name>/`) or `script` (prep script + manual raw
  download) — so `load()` knows where to find data and raises an actionable
  error for a missing script-only dataset (e.g. "download Home Credit from
  Kaggle, then run `prepare/home_credit.py`").
- **Prep scripts** (`benchmarks/datasets/prepare/<name>.py`). Deterministic
  transform: raw source → mononet-convention `train_<name>`/`test_<name>` with
  the declared monotone columns present, a fixed split (chronological where the
  source provides time, else stratified with a fixed seed), test never mixed into
  train. For LFS datasets the script documents how the committed file was
  produced; for script-only datasets the user runs it.
- **LFS layout & size.** Committed data lives under `benchmarks/data/**`, matched
  by a scoped `.gitattributes` filter (`benchmarks/data/** filter=lfs diff=lfs
  merge=lfs -text`) so no stray CSV elsewhere is swept in. Large tables are
  stored **gzip-compressed** (`.csv.gz`, read transparently by pandas) to respect
  GitHub's ~1 GB LFS quota (ACSIncome ~50–80 MB gz vs ~300 MB raw).
- **Manifest.** `manifest.toml` extends to carry, per file, `sha256` + source
  URL + license + hosting class; the loader verifies checksums via the existing
  `verify()`.

### 4.2 Devcontainers

Add `git-lfs` install + `git lfs install` to `shared/install_common_tools.sh` so
all five flavors (default, gpu-torch, gpu-jax, gpu-keras, proofs) can pull LFS
data. Document `git lfs pull` in CONTRIBUTING and note it as a CI prerequisite
for data-dependent jobs.

### 4.3 Max-size screen (`benchmarks/large_screen_run.py`)

Per dataset, run the existing standard search for **both arms** at full size —
deep (`depth ∈ {6, 10, 16}`) and shallow (`depth ∈ [1, 4]`), both
`mode="absolute"`, `residual=True` — then multi-seed refit + test on the
untouched test set, and compute `Δ = IQM(deep) − IQM(shallow)` with the
seed-bootstrap band by reusing `size_ladder_report.delta_by_n` at a single top
rung. Metric: accuracy (classification) / RMSE (regression, sign-normalized so
"deep better" ⇒ Δ > 0). Per-dataset budget (n_trials, test-seeds, n_splits,
batch) is added to the existing `_DATASET_PROTOCOL` table; the large anchor
(ACSIncome) gets a larger batch and capped trials for wall-time.

### 4.4 Gate (`benchmarks/_common/screen_gate.py`)

Pure, unit-tested: `gate(delta_lo, delta_point, margin) -> "ladder" | "standard"`.

- **`ladder`** iff `delta_lo > 0` **and** `delta_point >= margin`;
- else **`standard`**.
- Margin: **0.005** accuracy (classification); **1% of shallow RMSE**
  (regression).

### 4.5 Conditional outcomes

- **`ladder`** → run the full N-ladder for that dataset. Generalize the loan
  launcher (`loan_ladder_launch.py`, `loan_size_ladder_run.py` — already
  bundle-parametrized) to take `--dataset <name>`; each ladder is its own
  GPU-session deliverable per the loan-ladder RUNBOOK.
- **`standard`** → wire the dataset into the standard roster: add to
  `_ALL_DATASETS` (`search.py`), `METRIC` + `_ORDER` (`make_tables.py`), and a
  `_DATASET_PROTOCOL` entry. `DatasetSpec` is already present.

Both paths are mechanical, driven by the committed screen verdict.

### 4.6 Results / report / docs

- `benchmarks/results/screen/*.json` — one record per dataset `{name, n_full,
  deep_iqm, shallow_iqm, delta, delta_lo, delta_hi, margin, verdict}`.
- `benchmarks/_common/screen_report.py` — per-dataset table (n_full, deep/shallow
  IQM, Δ ± CI, margin, verdict) + a summary plot (Δ with CI per dataset, sorted,
  reference lines at 0 and the margin), rendered as **PNG and PDF** (mirrors the
  ladder plot: mathtext, colorblind-safe, no title, `bbox_inches="tight"`).
- `docs/benchmarks/large-dataset-screen.md` — screen results page (table + plot +
  which datasets advanced to ladders vs joined the standard set); wired into the
  benchmarks toctree; cross-linked from `deep-residual-accuracy.md` and
  `loan-size-ladder.md`.
- `benchmarks/RUNBOOK-large-screen.md` — GPU run + report procedure (mirrors
  `RUNBOOK-loan-ladder.md`).

## 5. Testing

- **`DatasetSpec` validation** — every `mono_increasing`/`mono_decreasing` name is
  a real column; the target exists.
- **Loader** — each LFS dataset loads to the expected shape; a script-only
  dataset with the raw file missing raises the documented, actionable error.
- **Prep scripts** — unit-tested on a tiny synthetic raw sample → correct
  mononet-convention output (deterministic split, monotone columns present, class
  ratio preserved, test untouched).
- **`gate()`** — pure-function boundary tests (`delta_lo` just ±0; `delta_point`
  just ±margin; regression sign-normalization).
- **Screen smoke** — tiny synthetic bundle, both arms, ≤2 trials / 1 seed →
  asserts a record with a finite Δ and a valid verdict. Fast, CI-cheap, no real
  download.
- **Manifest** — sha256 verification test on a committed fixture.
- Existing suites stay green; `uv run mypy` clean across all backends;
  `sphinx-build -W` clean (new page + committed plot asset resolve). git-lfs
  presence documented; CI `git lfs pull` note for data-dependent jobs.

## 6. Scope split

- **Landed in this PR (plumbing + screen):** the `DatasetSpec` entries, data
  plumbing (source descriptor, prep scripts, LFS layout, manifest), devcontainer
  git-lfs, the screen run script, the gate, the report + docs page skeleton +
  RUNBOOK, and all tests. Mergeable with LFS data committed for redistributable
  datasets and smoke-level screen numbers.
- **GPU session (per RUNBOOK):** the real max-size screen across the roster;
  fills the report table/plot and commits the screen result JSONs and the gate
  verdicts.
- **Downstream (verdict-driven):** ladders for datasets that pass the gate;
  standard-roster wiring for those that do not — each tracked separately.

## 7. Non-goals / out of scope

- Learning-to-rank (Phase 2) and curve-regression (Phase 3) — separate specs.
- Fannie/Freddie manual-download, LHCb reconstruction, synthetic physics —
  backlog (program note).
- Cross-dataset significance machinery (Friedman/Nemenyi).
- Any `mononet` package, kernel, or `model_builder` change.

## 8. Open items

- Give Me Some Credit hosting: confirm the OpenML mirror's license permits
  re-hosting; if not, downgrade to script-only. Decide at implementation.
- ACSIncome scope: which survey year(s)/horizon to commit as the canonical
  "full" rung (a single 2018 1-Year US-wide pull vs multi-year pooling). Pick the
  smallest that anchors the curve, documented in the prep script.
- Whether Adult is redundant with ACSIncome or a valuable small within-domain
  contrast — keep unless the screen shows them degenerate.
