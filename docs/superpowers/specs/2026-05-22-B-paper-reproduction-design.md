# Sub-project B — Paper reproduction benchmarks

**Date:** 2026-05-22
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Depends on:** [Sub-project A](2026-06-27-A-core-algorithm-and-backends-design.md) (locked public API)
**Paper:** Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>

## 1. Goals & non-goals

### Goals

- Reproduce **Table 1** of the paper (COMPAS, Blog Feedback, Loan Defaulter) with `mononet.keras.MonoDense` (the paper's reference framework was Keras/TF) and verify the result with the equivalent `mononet.torch` and `mononet.jax` models.
- Reproduce **Table 2** of the paper (Auto MPG, Heart Disease) on the same three backends.
- Match the paper's reported metrics within reasonable statistical uncertainty (mean ± std across 10 seeds, reporting the same top-5-of-10 statistic the paper uses).
- Provide a reusable benchmark harness — datasets, configs, runners, results aggregation, table emission — that Sub-project C inherits.
- Render the benchmark results into the Sphinx docs site as the "Benchmarks → Reproducing the paper" section.
- Provide one notebook per dataset, executable end-to-end, committed with outputs (per parent spec §8).

### Non-goals

- No re-running of the paper's Bayesian hyperparameter search. We use the hyperparameters the paper reports (and where the paper is silent, the ones in <https://github.com/airtai/monotonic-nn>, the author's original codebase).
- No new datasets (those are Sub-project C).
- No ablations of activation choice, depth, width, or split — these are also Sub-project C.
- No baselines beyond what the paper compares against. We do not invent new comparators.
- No GPU CI for the benchmarks — execution is manual and local per parent spec §8.

## 2. Datasets and where they live

| Dataset | Type | Features | Mono features | Source | Notes |
|---|---|---|---|---|---|
| COMPAS | binary classification | 13 | 4 | ProPublica github (`compas-analysis`) | Use the *exact* train/test split from Liu et al. 2020 |
| Blog Feedback | regression | 276 | 8 | UCI ML repo | Same split as Liu et al. 2020 |
| Loan Defaulter | binary classification | 28 | 5 | Kaggle (Lending Club) | Requires Kaggle CLI auth; ~500k rows |
| Auto MPG | regression | 7 | 3 | UCI ML repo | Split per Sivaraman et al. 2020 |
| Heart Disease | binary classification | 13 | 2 | UCI ML repo | Split per Sivaraman et al. 2020 |

Each dataset has a loader in `benchmarks/datasets/<dataset>.py`:

```python
def load(cache_dir: Path) -> DatasetBundle:
    """Download if absent (with checksum), preprocess to canonical
    (X_train, y_train, X_test, y_test, monotonicity_mask) and cache."""
```

`DatasetBundle` is a frozen dataclass under `benchmarks/_common/`:

```python
@dataclass(frozen=True, slots=True)
class DatasetBundle:
    name: str
    task: Literal["binary_classification", "regression"]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    monotonicity: MonotonicityMask
    feature_names: tuple[str, ...]
    metadata: dict[str, str]          # citation, source URL, preprocessing notes
```

Downloads cache to `~/.cache/mononet/datasets/<dataset>/` keyed by SHA-256 of the source archive. Loaders verify checksums on every call and re-download on mismatch. Loan Defaulter, because of its Kaggle gate, has an extra branch that reads from a user-provided path if Kaggle CLI is not configured (documented in `benchmarks/datasets/README.md`).

## 3. Repo layout

This sub-project does **not** live inside the `mononet/` package — it ships as a separate top-level `benchmarks/` tree. The published wheel must stay small; benchmark code, fixtures, and notebook scaffolding go in the repo but not in the distribution.

```
benchmarks/
├── README.md                   # how to run; cache dir; dataset auth notes
├── __init__.py
├── _common/
│   ├── bundle.py               # DatasetBundle
│   ├── runner.py               # generic train/eval loop
│   ├── results.py              # ResultRow, JSON IO, table rendering
│   └── seeds.py                # deterministic seed scheduling
├── datasets/
│   ├── compas.py
│   ├── blog_feedback.py
│   ├── loan_defaulter.py
│   ├── auto_mpg.py
│   └── heart_disease.py
├── configs/                    # one TOML per (dataset, backend)
│   ├── compas-keras.toml
│   ├── compas-torch.toml
│   ├── compas-jax.toml
│   └── ...
├── baselines/
│   ├── xgboost.py              # comparator from Table 1
│   ├── isotonic.py             # comparator from Table 1
│   ├── non_neg_dnn.py          # trivial: our MonoMLP with no concave/saturated buckets
│   ├── min_max_net.py          # small re-implementation (~80 LOC)
│   └── dln.py                  # uses tensorflow-lattice (optional install)
└── run.py                      # CLI: `python -m benchmarks.run --dataset compas --backend torch`
```

Notebooks live in `docs/benchmarks/paper-reproduction/`:

```
docs/benchmarks/paper-reproduction/
├── index.md
├── compas.ipynb
├── blog-feedback.ipynb
├── loan-defaulter.ipynb
├── auto-mpg.ipynb
├── heart-disease.ipynb
└── tables.ipynb                # aggregated Tables 1 & 2
```

Notebooks shell out to `benchmarks.run` for execution and visualize the JSON result files. This keeps the heavy logic testable in plain Python.

## 4. The benchmark runner

```python
@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    dataset:        str                          # "compas"
    backend:        Literal["torch", "jax", "keras"]
    model:          Literal["mono_mlp", "mono_feature_block"]
    hidden_features: tuple[int, ...]
    activation:     str
    activation_split: tuple[int, int, int] | Literal["thirds"]
    init:           InitSpec
    optimizer:      OptimizerSpec                # {name, lr, weight_decay, ...}
    batch_size:     int
    epochs:         int
    early_stopping: EarlyStoppingSpec | None
    seeds:          tuple[int, ...]              # e.g. (0, 1, 2, ..., 9)
    metrics:        tuple[Literal["accuracy", "rmse", "auc", "mse"], ...]

def run(cfg: BenchmarkConfig, *, cache_dir: Path) -> list[ResultRow]:
    bundle = datasets.load(cfg.dataset, cache_dir)
    rows = []
    for seed in cfg.seeds:
        model  = build_model(cfg, bundle, seed)
        history = train(model, bundle, cfg, seed)
        scores  = evaluate(model, bundle, cfg.metrics)
        rows.append(ResultRow.from_run(cfg, seed, scores, history))
    return rows
```

`ResultRow` serializes to one JSON-Lines record per `(config, seed)`. Tables 1 / 2 are emitted by `results.py:render_table(rows, schema)` which pivots on `(dataset, method)` and reports `mean ± std` over the top-5-of-10 seeds (matching the paper's reporting convention).

## 5. Hyperparameter sources

Reproducibility requires *exact* hyperparameter values. The paper's appendix and the [airtai/monotonic-nn](https://github.com/airtai/monotonic-nn) reference repo are the canonical sources. The reference repo's hyperparameters are committed into `benchmarks/configs/*.toml` verbatim, with a header comment naming the source commit hash and the paper section.

Example: `benchmarks/configs/compas-keras.toml`:

```toml
# Source: airtai/monotonic-nn @ <commit hash> — configs/compas.yml
# Paper section: §4 + supplementary §C.1
[dataset]
name = "compas"

[model]
backend = "keras"
type = "mono_mlp"
hidden_features = [16, 16, 16]
activation = "elu"
activation_split = "thirds"

[init]
scheme = "glorot_uniform"
seed = null

[optimizer]
name = "adam"
lr = 1e-3

[run]
batch_size = 64
epochs = 200
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
metrics = ["accuracy"]

[early_stopping]
patience = 20
monitor = "val_loss"
```

Identical TOMLs are provided for `compas-torch.toml` and `compas-jax.toml`. The three backends share everything except the `backend` field. If the per-backend runs disagree on the mean score beyond a tolerance (default: max(0.5 std, 0.2%)), we treat that as a regression and fail CI.

## 6. Baselines

| Baseline | How we get it | Risk |
|---|---|---|
| Isotonic regression | `sklearn.isotonic` | Trivial. |
| XGBoost | `xgboost.XGBClassifier` / `XGBRegressor` | Standard. |
| Non-Neg-DNN | Our `MonoMLP` with `activation_split=(m, 0, 0)` (convex-only) | Free with Sub-project A. |
| Crystal (Milani Fard et al., 2016) | `tensorflow-lattice` if available; otherwise quote the paper's number | TFL is heavyweight. Make this an optional `bench-tfl` extra; if not installed, fall back to "quoted from paper". |
| DLN (You et al., 2017) | Same as Crystal — `tensorflow-lattice` | Same. |
| Min-Max Net (Daniels & Velikova, 2010) | Re-implement in ~80 LOC in `baselines/min_max_net.py` | Small, well-defined. |
| Certified (Liu et al., 2020) | Quote from paper; do not re-run (MILP solver dependency) | Significant infra cost; quoted result OK. |
| COMET (Sivaraman et al., 2020) | Quote from paper (SMT solver dependency) | Same. |
| `airtai/monotonic-nn` (original code) | Manual install via `--no-deps` (see parent spec §5) | Used only for numerical sanity-check (output match) on one dataset, not for benchmarking. |

The "quoted from paper" rows are clearly tagged in the table:

```
COMPAS test accuracy:
  Ours (mononet.keras, this run): 69.1 ± 0.2%
  Paper Table 1, our method:      69.2 ± 0.2%       [quoted]
  Paper Table 1, Certified:       68.8 ± 0.2%       [quoted]
```

This makes it transparent which numbers came from our infrastructure and which were copy-pasted from the paper.

## 7. Cross-backend agreement

For each of the five datasets we run all three backends (keras, torch, jax) with the same TOML modulo the `backend` field. We assert in `tables.ipynb`:

```
| dataset    | metric  | keras-mean | torch-mean | jax-mean | max-pairwise-gap |
| compas     | acc     | 0.691      | 0.689      | 0.690    | 0.002            |
| auto_mpg   | mse     | 8.37       | 8.41       | 8.39     | 0.04             |
| ...
```

`max-pairwise-gap > 0.5 std` is treated as a soft alarm (manually investigated). The Sub-project A equivalence tests already guarantee per-instance backend agreement on synthetic inputs; this is the end-to-end-training corroboration.

## 8. Execution and CI

Following parent spec §8:

- Notebook execution is **manual**, on a `gpu-*` devcontainer when GPU is faster.
- One `tools/execute-benchmarks.sh` helper runs `nbconvert --execute --inplace` over `docs/benchmarks/paper-reproduction/*.ipynb`.
- Notebooks are committed with outputs; Sphinx renders the committed outputs (`execute: false` in `myst-nb` config).
- CI does **not** execute notebooks. Instead, CI runs a small `tests/benchmarks/test_loaders.py` that verifies dataset loaders work end-to-end on a sampled subset (e.g. 1000 rows of Blog Feedback), and that `benchmarks.run --dataset auto_mpg --seeds 0 --epochs 2` completes for each backend (smoke test). Real benchmark numbers come from the maintainer's manual run.

Pre-release ritual: run all 15 notebook executions (5 datasets × 3 backends), eyeball Tables 1 & 2 in `tables.ipynb`, commit the refreshed outputs, then tag.

## 9. What the docs site shows

Under "Benchmarks → Reproducing the paper":

- `index.md` — one paragraph explaining what's reproduced, link to paper, table of contents.
- One page per dataset — embedded notebook with loading, training, metric extraction.
- `tables.md` — the rendered Tables 1 and 2, side-by-side with the paper's published numbers. This is the page paper readers land on.

## 10. Open items

- Loan Defaulter's Kaggle license — confirm it allows redistribution of the preprocessed split. If not, ship loader only (downloads on user machine after `kaggle datasets download`).
- `tensorflow-lattice` install path on Python 3.13 — may not have wheels yet. If broken, drop the Crystal + DLN re-runs and rely entirely on the paper's quoted numbers for those rows.
- Whether to include a CPU-only execution path for Loan Defaulter — full dataset is ~500k rows, may be slow on the CPU devcontainer. Decide based on profiling; potentially down-sample for the "smoke" notebook but full-size for the headline number.
- Exact "top 5 of 10" statistic: the paper says "we run the experiments ten times after finding the optimal hyperparameters and report the mean and standard deviation of the best five results". Confirm the criterion for "best" (test metric per seed? validation? lowest train loss?). Read airtai/monotonic-nn source to settle.

## 11. What is intentionally NOT in this sub-project

- **Bayesian hyperparameter optimization.** The paper used GP-based BO; we use the discovered hyperparameters and trust them.
- **Newer datasets / extended ablations / activation-choice sweeps.** All belong to Sub-project C.
- **Training-loop helpers shipped in the published wheel.** The runner is benchmark-internal infrastructure, not public API.
