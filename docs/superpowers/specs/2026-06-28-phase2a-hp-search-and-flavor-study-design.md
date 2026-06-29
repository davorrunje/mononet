# Phase 2a — Hyperparameter Search Engine + Flavor Study — Design

**Date:** 2026-06-28
**Status:** Approved
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Builds on:** [Benchmark foundation + reproduction (Phase 1)](2026-06-28-benchmark-foundation-and-reproduction-design.md) (PR #51, merged) — the `benchmarks/` harness, `DatasetBundle`, `BenchmarkConfig`/`run(cfg)`, loaders, results aggregation.
**Paper:** Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>

> Phase 2 was sliced into **2a** (this spec: HP-search engine + flavor study on the
> five paper datasets) and **2b** (a later spec: physics/medicine datasets, reusing
> this engine). The physics/medicine datasets, `activation` as a search axis, new
> layer variants, and heavy baselines are **out of scope here**.

## 1. Goals & non-goals

### Goals
- A shared **Optuna-based search engine** (Python, in `benchmarks/`) that drives the
  existing sweep-ready `run(cfg)` to tune hyperparameters, usable across all three
  backends through one code path.
- A **fair flavor study**: tune each of the four flavors `{switch, absolute} ×
  {plain, residual}` independently per dataset, then compare them at their own tuned
  best on the five paper datasets.
- A **validation split** carved from train (search never touches test), added as a
  helper, not by changing `DatasetBundle`.
- Searches launched from **notebooks** (`benchmarks/notebooks/`, repo-only, manual),
  with a **rendered summary notebook** in `docs/` showing the flavor-comparison
  tables/plots from committed results.
- Unit-testable engine (CI runs a tiny synthetic search; real searches are manual).

### Non-goals (deferred)
- Physics/medicine datasets (Phase 2b).
- `activation` as a search axis (held at `elu`); new layer variants; heavy baselines
  beyond the existing XGBoost.
- Running real searches in CI (manual maintainer runs, like the Phase-1 headline numbers).
- Any change to the shipped `mononet` package or the `DatasetBundle` contract.

## 2. New requirement: validation split (helper, not a bundle field)

`DatasetBundle` (Phase 1, frozen) exposes only train/test. HP search must optimize a
signal that is not the test set. We add a **helper** rather than a bundle field, so the
Phase-1 contract, loaders, and tests stay untouched:

`benchmarks/_common/splits.py`:

```python
def train_val_split(
    bundle: DatasetBundle, *, frac: float = 0.2, seed: int, stratify: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split bundle.X_train/y_train into (X_tr, y_tr, X_val, y_val).

    Stratified for binary_classification by default; deterministic given seed.
    Does NOT touch X_test/y_test.
    """
```

- `stratify` defaults to `True` for `task == "binary_classification"`, else `False`.
- Deterministic for a given `seed` (uses `numpy`/`sklearn.model_selection.train_test_split`).
- The test set is never read by the search path.

## 3. The search engine

`benchmarks/_common/search_spaces.py` — per-flavor search space:

```python
def suggest_config(
    trial: optuna.Trial, *, dataset: str, mode: str, residual: bool,
) -> BenchmarkConfig:
    """Sample a BenchmarkConfig for (dataset, flavor) from the trial."""
```

Searched axes (ranges documented in the module, seeded by Optuna's sampler):

| Axis | Range | Notes |
|---|---|---|
| `lr` | log-uniform `[1e-4, 1e-1]` | |
| `weight_decay` | uniform `[0.0, 0.2]` | |
| `dropout` | uniform `[0.0, 0.5]` | |
| `lr_decay` | uniform `[0.85, 1.0]` | per-epoch multiplier |
| `batch_size` | categorical `{8, 16, 32, 64, 128, 256}` | |
| `depth` | int `[1, 4]` | |
| `width` | categorical `{8, 16, 21, 32, 64}` | |
| `convex_fraction` | uniform `[0.0, 1.0]` | **only when `mode == "absolute"`** |
| `activation` | fixed `"elu"` | not searched in 2a |
| `embed_hidden` | derived from `width` (mirror) | not independently searched |

`benchmarks/_common/search.py`:

```python
@dataclass(frozen=True, slots=True)
class StudyResult:
    dataset: str
    flavor: str                  # "switch-plain", "absolute-residual", ...
    best_params: dict[str, Any]
    best_value: float            # best VALIDATION metric
    n_trials: int

def search(
    dataset: str, *, mode: str, residual: bool, backend: str,
    n_trials: int = 50, seed: int = 0, storage: str | None = None,
) -> StudyResult:
    """Run an Optuna study tuning (dataset, flavor) HPs on the validation split."""
```

- Direction: minimize for `mse`/`rmse`, maximize for `accuracy` (from the dataset task).
- Each trial: `train_val_split(bundle, seed=seed)`, build the sampled `BenchmarkConfig`,
  train on `(X_tr, y_tr)` for the config's epochs (one seed per trial for speed),
  evaluate on `(X_val, y_val)`, report the metric to Optuna.
- **Median pruner** on the per-epoch validation metric to kill weak trials early.
- `storage` (an Optuna SQLite URL) is optional; when set, studies persist/resume.

## 4. Study matrix and final evaluation

For each of the **5 datasets × 4 flavors = 20** `(dataset, flavor)` pairs:

1. **Search** (`search(...)`) → best HPs on validation.
2. **Final eval** (separate from search): take the best HPs, **refit on the full train
   split** (train + the held-out val), run **10 seeds**, report the **test** metric as
   **mean ± std of best-5-of-10** (matching Phase-1 reporting).
3. Write `benchmarks/results/phase2/<dataset>-<flavor>.json` (committed): best HPs,
   `best_value` (val), and the final test aggregate.

Flavor strings are `"<mode>-<plain|residual>"`, e.g. `"switch-plain"`, `"absolute-residual"`.

## 5. Notebooks & results

- `benchmarks/notebooks/search-<dataset>.ipynb` (one per dataset: auto, heart, compas,
  loan, blog) — **repo-only, NOT Sphinx-rendered**, run manually (GPU). Each loads the
  bundle, runs `search()` for the 4 flavors, runs the final eval, and writes the
  per-flavor result JSON. Committed without heavy outputs.
- `docs/benchmarks/flavor-comparison.ipynb` — **rendered**. Reads the committed
  `benchmarks/results/phase2/*.json` and renders, per dataset, a table of the 4 flavors
  at their tuned best vs the paper-quoted number vs XGBoost, plus a short "which flavor
  wins where" synthesis. Wired into the docs toctree.
- **Git-ignored:** Optuna SQLite study DBs and raw per-trial logs (`benchmarks/results/phase2/*.db`,
  `*.jsonl`). Committed: the distilled `<dataset>-<flavor>.json`.

## 6. Repo layout additions

```
benchmarks/
├── _common/
│   ├── splits.py            # train_val_split
│   ├── search.py            # Optuna wrapper: search() -> StudyResult
│   └── search_spaces.py     # suggest_config() per flavor
├── notebooks/
│   ├── search-auto.ipynb
│   ├── search-heart.ipynb
│   ├── search-compas.ipynb
│   ├── search-loan.ipynb
│   └── search-blog.ipynb
└── results/phase2/
    ├── .gitignore           # *.db, *.jsonl
    └── <dataset>-<flavor>.json   # committed best HPs + test aggregate (maintainer run)
docs/benchmarks/flavor-comparison.ipynb
tests/benchmarks/test_splits.py
tests/benchmarks/test_search.py
```

`pyproject.toml` `bench` group gains `optuna` (`==`-pinned per repo policy). `optuna` is
import-guarded in tests via `pytest.importorskip("optuna")`.

## 7. Testing / CI

- `test_splits.py`: shapes sum to the train size; stratification preserves class balance
  for binary tasks; determinism for a fixed seed; the test set is never returned/leaked.
- `test_search.py`: a **2-trial** `search()` on a tiny **synthetic** bundle (one backend
  via `importorskip`) returns a `StudyResult` with a finite `best_value` and
  `n_trials == 2`; the sampled config respects the conditional `convex_fraction`
  (present for absolute, absent/ignored for switch).
- CI runs only these fast synthetic tests. Real 20-study searches and the committed
  result JSONs are **manual maintainer runs**, documented in `benchmarks/README.md`.
- The rendered summary notebook builds under `./tools/build-docs.sh` with no executed
  outputs (reads committed JSON; `execute: off`).

## 8. Acceptance

- `train_val_split` and `search()` exist with the signatures above; engine is
  backend-agnostic (drives `run(cfg)`).
- The 20-study matrix is producible from the per-dataset search notebooks; each emits a
  committed `<dataset>-<flavor>.json` with best HPs + a test aggregate.
- `flavor-comparison.ipynb` renders the comparison from committed JSON and builds clean
  under strict docs.
- CI: `test_splits.py` + `test_search.py` pass on the active backend (synthetic, fast);
  ruff + mypy + static-analysis + pre-commit clean; no real search in CI.
- `optuna` added to the `bench` group; no `mononet` package change; `DatasetBundle`
  unchanged.

## 9. Open items

- **Search budget per dataset.** Default 50 trials; Loan/Blog (large) may warrant fewer
  trials or a capped per-trial epoch budget — document the per-dataset choice in each
  notebook header.
- **Pruner aggressiveness.** Median pruner defaults; revisit if it kills promising late
  bloomers on the small datasets.
- **convex_fraction extremes.** `0.0`/`1.0` give pure concave/convex; confirm the
  absolute-mode builder handles the boundaries (Phase-1 default was `0.5`).
- **Final-eval refit cost.** Refitting 4 flavors × 10 seeds × 5 datasets after search;
  acceptable on the small datasets, possibly slow for Loan — down-scale seeds there if
  needed (document).
