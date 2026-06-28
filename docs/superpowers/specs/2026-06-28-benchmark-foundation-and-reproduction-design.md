# Benchmark Foundation + Paper Reproduction — Design

**Date:** 2026-06-28
**Status:** Approved
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Depends on:** [Sub-project A](2026-06-27-A-core-algorithm-and-backends-design.md) (locked public API)
**Paper:** Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>
**Data:** <https://zenodo.org/records/7968969> (authors' preprocessed CSVs, CC-BY-4.0)
**Reference code/HPs:** <https://github.com/airtai/monotonic-nn/tree/main/nbs/experiments>

> **Supersedes** the stale `2026-05-22-B-paper-reproduction-design.md`, which predates
> the final Sub-project A API (it references the dropped `MonoMLP`/`MonoFeatureBlock`
> classes and the 3-class `(s̆,ŝ,s̃)` activation split). This is "Sub-project B,
> refreshed." The big hyperparameter searches, flavor sweeps, and new physics/medicine
> datasets are **Phase 2** (a follow-up spec), not this one.

## 1. Goals & non-goals

### Goals
- A **data-download script** that fetches the five preprocessed paper datasets from the
  Zenodo deposit into a git-ignored cache (never committed to git or LFS).
- A reusable **benchmark harness** — `DatasetBundle`, loaders, model builder, runner,
  results aggregation — built on the real Sub-project A API, and designed so a Phase-2
  sweep/HP-search layer can drive it without redesign.
- **Reproduce the paper's reported metrics** on the five datasets (COMPAS, Blog Feedback,
  Loan Defaulter, Auto MPG, Heart Disease) using mononet, comparing **four flavors** —
  `{switch, absolute} × {plain, residual}` — across all three backends (torch, jax, keras).
- A **cross-backend agreement** check per flavor (end-to-end training corroboration of the
  Sub-project A equivalence tests).
- Notebooks (committed with outputs) and a Sphinx "Benchmarks → Reproducing the paper"
  section.

### Non-goals (deferred to Phase 2)
- Hyperparameter **search** (Optuna/keras-tuner-style). Phase 1 uses the *already-tuned*
  hyperparameters from the airtai notebooks. The runner is built sweep-ready; the search
  engine is not built here.
- **New datasets** (physics/medicine candidates: Energy Efficiency, Concrete,
  Superconductivity, Diabetes-130, Pima, Breast-Cancer-Wisconsin).
- **Heavy baselines** (Min-Max net, tensorflow-lattice DLN/Crystal, isotonic). Phase 1
  ships XGBoost + the paper's quoted numbers only.
- Any change to the shipped `mononet` package. The harness is repo-only, not in the wheel.

## 2. Data acquisition

`benchmarks/datasets/download.py` (also runnable as `python -m benchmarks.datasets.download`):

- Downloads the 10 CSVs from `https://zenodo.org/records/7968969/files/<name>.csv?download=1`
  (`{train,test}_{auto,blog,compas,heart,loan}.csv`).
- Default destination **`~/.cache/mononet/datasets/`**, overridable via `--dest` / env
  `MONONET_DATA_DIR`. Never written inside the repo tree; never committed (git or LFS).
- Verifies each file against a committed **SHA-256 manifest**
  (`benchmarks/datasets/manifest.toml`); re-downloads on mismatch, fails loudly on a
  second mismatch.
- Records the CC-BY-4.0 attribution and the Zenodo DOI (`10.5281/zenodo.7968969`) in the
  manifest header.
- Idempotent: skips files already present with a matching checksum.

## 3. DatasetBundle and loaders

`benchmarks/_common/bundle.py`:

```python
@dataclass(frozen=True, slots=True)
class DatasetBundle:
    name: str
    task: Literal["binary_classification", "regression"]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    mono_increasing: tuple[int, ...]   # column indices, non-decreasing
    mono_decreasing: tuple[int, ...]   # column indices, non-increasing
    feature_names: tuple[str, ...]
    metadata: dict[str, str]           # source URL, DOI, preprocessing notes
```

Non-monotone features are exactly the columns in neither tuple. Each loader in
`benchmarks/datasets/<name>.py` reads the cached CSVs, applies the column/target split
the authors used, and fills the monotonicity tuples from the paper / airtai notebooks:

| Dataset | Task | Features | Monotone features (direction) |
|---|---|---|---|
| Auto MPG | regression (MSE) | 7 | weight ↓, displacement ↓, horsepower ↓ |
| Heart Disease | binary (accuracy) | 13 | 2 per the Heart.ipynb indicators |
| COMPAS | binary (accuracy) | 13 | 4 per Compas.ipynb |
| Blog Feedback | regression (RMSE) | 276 | 8 per Blog.ipynb |
| Loan Defaulter | binary (accuracy) | 28 | 5 per Loan.ipynb |

Exact monotone-column indices and directions are transcribed verbatim from the
corresponding airtai notebook, with a source comment in each loader. A
`benchmarks/datasets/registry.py` maps name → loader.

## 4. Model builder — four flavors + partial monotonicity

`benchmarks/_common/model_builder.py` builds a native `Sequential`/`Module`/`Layer` per
backend. Because these datasets are mostly **non-monotone** and mononet has no `t=0`
path, the builder uses the **embedding-composition** architecture (the faithful,
interaction-preserving stand-in for the paper's mixed-`t` `MonoDense`):

```
non-monotone columns ─► unconstrained MLP (framework-native) ─► embedding e
monotone columns m   ─► MonoInput(signs) ────────────────────────────┐
                                       concat([m, e]) ─► monotone stack ─► head
```

- Output is **non-decreasing in m** (m enters a non-decreasing stack directly) and
  unconstrained in the raw non-monotone columns (`e` is a free function of them);
  interactions occur inside the monotone stack.
- `MonoInput(mono_increasing ∪ -mono_decreasing)` applies the sign flips so the stack
  sees only non-decreasing inputs.
- The **monotone stack** is where the four flavors live:
  - **plain:** `MonoLinear/MonoDense(width, mode, convex_fraction=0.5, activation)` ×depth
  - **residual:** `MonoResidual(width, width, mode, activation)` ×depth (after a width projection)
  - `mode ∈ {switch, absolute}`.
- **Head:** linear (regression) or sigmoid (binary) — sigmoid preserves monotonicity.
- **Dropout** (between layers) and **weight_decay** / **LR decay** are supported because the
  airtai HPs use them. Dropout is identity at inference, so monotonicity holds at eval.
- If a dataset had zero non-monotone columns the embedding branch is omitted; none of the
  five are fully monotone, so the branch is always present here.

The unconstrained embedding MLP is identical across the four flavors (only the monotone
stack changes), so flavor differences are attributable to the monotone construction.

## 5. Runner, config, metrics

`benchmarks/_common/runner.py`:

```python
@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    dataset:        str
    backend:        Literal["torch", "jax", "keras"]
    mode:           Literal["switch", "absolute"]
    residual:       bool
    depth:          int
    width:          int
    activation:     str                       # "elu", "relu", ...
    convex_fraction: float                    # 0.5 default
    embed_hidden:   tuple[int, ...]           # unconstrained branch shape
    dropout:        float
    optimizer:      OptimizerSpec             # name, lr, weight_decay
    lr_decay:       float | None
    batch_size:     int
    epochs:         int
    early_stopping: EarlyStoppingSpec | None
    seeds:          tuple[int, ...]
    metrics:        tuple[Literal["accuracy", "rmse", "mse"], ...]

def run(cfg, *, cache_dir) -> list[ResultRow]: ...
```

`ResultRow` serializes one JSON-Lines record per `(config, seed)`. The headline statistic
matches the paper: **mean ± std of the best 5 of 10 seeds**; the plain 10-seed mean/std is
also recorded. (The airtai notebooks report mean/std over 3 runs of an already-tuned
config; we run 10 seeds and report the paper's best-5-of-10 to align with Table 1/2.)

`benchmarks/run.py` CLI:
`python -m benchmarks.run --dataset auto --backend torch --mode switch --residual`.

## 6. Reproduction targets & acceptance

For each dataset, render a table: rows = `{paper [quoted], 4 mononet flavors, XGBoost}`,
columns = the backends (or a primary backend with a cross-backend agreement sub-table).
Per-dataset headline metric: COMPAS/Loan/Heart = accuracy, Auto MPG = MSE, Blog = RMSE.

**Acceptance:**
- Each mononet flavor lands **within statistical noise** of the paper's quoted number, or
  the gap is **documented honestly** (we do not bit-reproduce the 3-class method, and the
  embedding-composition differs from the paper's mixed-`t` parametrization — a flavor may
  legitimately differ; we report, we don't fudge).
- Cross-backend pairwise gap per flavor ≤ `max(0.5·std, 0.2%)`; larger → flagged and
  investigated (soft alarm, not auto-fail, since training is stochastic).
- Auto MPG is the calibration anchor: the airtai AutoMPG notebook reports MSE 8.371 ±
  0.084 with 2×21 ELU; we expect a mononet flavor in that neighborhood.

## 7. Baselines (Phase 1)

- **XGBoost** (`xgboost.XGBClassifier`/`XGBRegressor`) — standard, easy, per backend-agnostic.
- **Paper-quoted** comparator rows (Certified, COMET, DLN, Min-Max, etc.), tagged
  `[quoted]` and never re-run in Phase 1.

## 8. Hyperparameters

Per-dataset, transcribed verbatim from the airtai notebooks into
`benchmarks/configs/<dataset>.toml`, with a header naming the source notebook and commit.
Worked example — Auto MPG (from `AutoMPG.ipynb`):

```toml
# Source: airtai/monotonic-nn nbs/experiments/AutoMPG.ipynb
[model]
depth = 2
width = 21
activation = "elu"
convex_fraction = 0.5
[train]
optimizer = "adam"
lr = 0.073407
weight_decay = 0.058583
dropout = 0.157718
lr_decay = 0.887923
batch_size = 16
epochs = 50
seeds = [0,1,2,3,4,5,6,7,8,9]
metrics = ["mse"]
```

The other four datasets' configs are transcribed from `Blog.ipynb`, `Compas.ipynb`,
`Heart.ipynb`, `Loan.ipynb` during implementation. A config carries the dataset-level HPs;
`mode`/`residual`/`backend` are supplied per run so one config yields the four-flavor ×
three-backend matrix.

## 9. Phase-2 seam (designed, not built)

`BenchmarkConfig` is a frozen dataclass with a `.replace(**overrides)` helper, and `run()`
takes one fully-specified config. A Phase-2 `sweep.py` (likely Optuna) will iterate
configs over search spaces and call `run()` unchanged. No search code, search spaces, or
new-dataset loaders are written in Phase 1.

## 10. Repo layout

```
benchmarks/
├── README.md
├── __init__.py
├── _common/{bundle,model_builder,runner,results,seeds}.py
├── datasets/{__init__,registry,download,manifest.toml,auto,heart,compas,blog,loan}.py
├── configs/{auto,heart,compas,blog,loan}.toml
├── baselines/xgboost.py
└── run.py
docs/benchmarks/paper-reproduction/{index,auto-mpg,heart-disease,compas,blog-feedback,loan-defaulter,tables}.ipynb
tests/benchmarks/test_smoke.py
```

The `bench` dependency group (already in `pyproject.toml`: scikit-learn, pandas,
matplotlib) gains `xgboost` and `requests` (or stdlib `urllib` for the downloader — prefer
stdlib to avoid a new dep). Notebooks shell out to `benchmarks.run` and visualize the JSON.

## 11. CI + docs

- **CI smoke test** (`tests/benchmarks/test_smoke.py`): builds each backend × one flavor on
  a tiny **synthetic** in-memory bundle (no network, no Zenodo dependency in CI) and runs 2
  epochs; asserts the model trains and emits a `ResultRow`. Dataset loaders are tested
  against a committed ~50-row fixture slice, not the full cache.
- CI does **not** download the Zenodo data or execute notebooks.
- Notebooks committed with outputs; executed manually via `tools/execute-benchmarks.sh`
  (`nbconvert --execute --inplace`); Sphinx renders committed outputs (`execute: false`).
- Aggregated headline results are committed as a small JSON under
  `benchmarks/results/paper-reproduction.json` (for the docs tables); raw per-seed JSONL is
  git-ignored.

## 12. Open items

- Exact monotone-column indices/directions for Blog/Compas/Heart/Loan — transcribe from the
  airtai notebooks during implementation (Auto MPG is confirmed: weight/displacement/horsepower ↓).
- Loan Defaulter is ~80 MB train CSV; the full run may be slow on CPU. Down-sample only for
  the smoke fixture; the headline number uses the full split (manual run, possibly on a
  `gpu-*` devcontainer).
- The embedding-composition's unconstrained-branch shape (`embed_hidden`) is not in the
  paper; pick a sensible default (e.g. mirror the monotone width) and document it as a
  reproduction choice, not a paper value.
- Confirm whether the paper's headline used best-5-of-10 or the airtai 3-run mean; default
  to best-5-of-10 and note the discrepancy if numbers diverge.

## 13. Non-goals restated

HP search, flavor *sweeps* beyond the fixed four, new datasets, heavy baselines, and any
`mononet` package change are all out of scope for Phase 1.
