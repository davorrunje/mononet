# Sub-project C — Extended benchmarks and ablations

**Date:** 2026-05-22
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Depends on:** [Sub-project A](2026-05-22-A-core-algorithm-and-backends-design.md) (locked public API), [Sub-project B](2026-05-22-B-paper-reproduction-design.md) (the runner, loaders, and result schema)
**Paper:** Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>

## 1. Goals & non-goals

### Goals

- Extend the benchmark harness from Sub-project B with **post-paper datasets** that have natural monotonic features, to characterize how `mononet` behaves outside the paper's selection.
- Run a **design-space ablation** sweep: activation choice, activation split, depth, width, architecture type (1 vs. 2), init scheme. Produce plots that turn each axis into a curve.
- Provide a **scaling** view: training time and parameter count as a function of input dimensionality and number of monotonic features, comparing `mononet` against XGBoost and a vanilla unconstrained MLP.
- Make every result reproducible from a single config + seed list.
- Render results into the Sphinx docs site as "Benchmarks → Beyond the paper" and "Benchmarks → Ablations".

### Non-goals

- No new algorithm. We benchmark, we do not extend the construction.
- No re-running of Tables 1 / 2 (that's Sub-project B).
- No invertibility / flow benchmarks (that's Sub-project D — its experiments live in that sub-project, not here).
- No formal-method-based monotonicity certification (Certified / COMET style). Out of scope.
- No paid-API model comparisons (e.g. no calls to commercial monotonic-modeling services). The library license is non-commercial; we keep comparators in the same spirit.

## 2. Extended dataset list

Selected for: presence of a *natural* monotonic feature, publicly downloadable, license-compatible with non-commercial benchmarking, and representative of tabular ML production scenarios.

| Dataset | Type | Features | Mono features | Source | Why we include it |
|---|---|---|---|---|---|
| **UCI Adult** | binary classification | 14 | `education-num` (↑), `hours-per-week` (↑, contested), `age` (↑/↓ debated) | UCI ML repo | Canonical fairness benchmark; well-studied. Income vs. education is a textbook monotonic prior. |
| **FICO HELOC** | binary classification | 23 | `MSinceMostRecentDelq` (↓ risk), `NumSatisfactoryTrades` (↑) etc. | FICO Community (BFCC) | Public credit risk dataset with the most defensible monotonicity priors. Used in the interpretability literature. |
| **California Housing** | regression | 8 | `median_income` (↑), `housing_median_age` (mild) | sklearn built-in | Replaces deprecated Boston Housing. Trivial loader. |
| **Ames Housing** | regression | ~80 | several (`OverallQual`, `GrLivArea`, ...) | Kaggle (House Prices: Adv. Regression) | High-dimensional tabular; tests how many monotonic features the library handles gracefully. |
| **Diabetes 130-US Hospitals** | binary classification (readmission) | 50 | `time_in_hospital` (mild ↑ readmit), `num_lab_procedures` | UCI ML repo | Healthcare; well-known monotonic clinical priors. |
| **MovieLens-1M (rating prediction)** | regression | varies | `user_rating_count` (mild) | grouplens.org | Different domain (collaborative filtering with monotonic side-features); also stresses Type 2 architecture (Fig. 5). |
| **Higgs (subset, 1M rows)** | binary classification | 28 | none natively monotonic | UCI / OpenML | **Negative control** — a dataset where the prior offers no monotonic features; we expect `mononet` (with all-zero mask) to degenerate to standard MLP performance. Useful sanity check. |

Each loader follows the `DatasetBundle` contract from Sub-project B. Caching, checksums, and Kaggle-auth fallback all reuse Sub-project B's infrastructure.

A `benchmarks/datasets/__init__.py` registry maps dataset name → loader. Headline runs use the same `benchmarks.run` CLI:

```
python -m benchmarks.run --dataset adult --backend torch --config configs/adult-torch.toml
```

## 3. Ablation matrix

Each ablation is a Cartesian product across one dimension while holding all others at the per-dataset paper defaults. Output: one CSV + one PNG plot per ablation per dataset.

| Axis | Levels | Plotted as |
|---|---|---|
| **Base activation** ρ̆ | `relu`, `elu`, `selu`, `gelu`, `tanh`, `sigmoid`, `softplus` | bar chart of test metric, error bars over 10 seeds |
| **Activation split** s | `(m,0,0)` convex-only, `(0,m,0)` concave-only, `(0,0,m)` saturated-only, `(m/2, m/2, 0)`, `(m/3, m/3, m/3)`, `(0, m/2, m/2)` | line chart with split label on x-axis |
| **Depth** | 1, 2, 3, 4, 5, 6 hidden layers | line chart, x = depth |
| **Width** | 8, 16, 32, 64, 128, 256, 512 | line chart, x = width (log scale) |
| **Architecture type** | Type 1 (`MonoMLP`) vs. Type 2 (`MonoFeatureBlock`) | grouped bar chart |
| **Init scheme** | `glorot_uniform`, `he_normal`, `lecun_normal` | bar chart |

Total ablations: 6 axes × 4 datasets that exercise each axis well = ~24 runs per axis-dataset pair. With 10 seeds each, that's 240 trainings per axis-dataset pair — manageable in hours on a single GPU. We pick **one** representative dataset per ablation axis for the canonical reported plot (e.g. Adult for activation choice, Ames for depth/width, FICO HELOC for architecture type). Other datasets get a "small" version of the same ablation in an appendix notebook.

## 4. Scaling characterization

A separate study (`docs/benchmarks/scaling/scaling.ipynb`):

1. **Parameter efficiency** — for the 5 paper datasets + 7 extended datasets, plot `parameters vs. test metric` for `mononet`, XGBoost (max_depth ∈ {4, 6, 8}), unconstrained MLP (same shape as `MonoMLP`). Reproduces and extends the paper's "order-of-magnitude fewer parameters" claim.

2. **Training-time scaling** — fix architecture, vary `n_train ∈ {1k, 10k, 100k, 1M}` using Higgs (subset) and Loan Defaulter. Plot wallclock per epoch and convergence epochs. One curve per backend.

3. **Forward-pass throughput** — vary `batch_size` and `out_features`; plot samples/sec for `MonoLinear` vs. `nn.Linear` per backend. This characterizes the per-step overhead of `|.|_t` and the three-way activation split.

4. **Monotonicity-feature-count scaling** — synthetic data: a ground-truth monotonic function `f(x_1, ..., x_n)` with controllable monotonic-feature count. Plot test MSE vs. number of declared monotonic features, with `mononet` vs. unconstrained MLP. Shows the data-efficiency benefit of the inductive bias as a function of how many features carry it.

## 5. Repo layout additions

All additive to Sub-project B's structure:

```
benchmarks/
├── datasets/
│   ├── adult.py
│   ├── fico_heloc.py
│   ├── california_housing.py
│   ├── ames_housing.py
│   ├── diabetes_130us.py
│   ├── movielens_1m.py
│   └── higgs_subset.py
├── configs/
│   ├── adult-{torch,jax,keras}.toml
│   ├── fico-{...}.toml
│   ├── ablations/
│   │   ├── activation-adult.toml
│   │   ├── depth-ames.toml
│   │   ├── width-ames.toml
│   │   ├── split-adult.toml
│   │   ├── arch-fico.toml
│   │   └── init-adult.toml
│   └── scaling/
│       ├── params-efficiency.toml
│       ├── train-time-loan.toml
│       ├── throughput-torch.toml
│       └── mono-feature-count-synth.toml
├── synthetic.py                # synthetic monotonic functions for §4.4
└── sweep.py                    # ablation-aware runner; accepts a sweep TOML

docs/benchmarks/
├── beyond-paper/               # extended datasets
│   ├── adult.ipynb
│   ├── fico-heloc.ipynb
│   ├── california-housing.ipynb
│   ├── ames-housing.ipynb
│   ├── diabetes-130us.ipynb
│   ├── movielens-1m.ipynb
│   ├── higgs-subset.ipynb       # the negative control
│   └── summary.ipynb
├── ablations/
│   ├── activation.ipynb
│   ├── split.ipynb
│   ├── depth.ipynb
│   ├── width.ipynb
│   ├── architecture.ipynb
│   └── init.ipynb
└── scaling/
    ├── parameters.ipynb
    ├── train-time.ipynb
    ├── throughput.ipynb
    └── mono-feature-count.ipynb
```

## 6. The sweep runner

```python
@dataclass(frozen=True, slots=True)
class SweepAxis:
    name:   str                              # "activation"
    field:  str                              # path within BenchmarkConfig
    values: tuple[Any, ...]

def run_sweep(
    base_config: BenchmarkConfig,
    axis:        SweepAxis,
    cache_dir:   Path,
) -> SweepResult:
    rows = []
    for value in axis.values:
        cfg = base_config.replace(**{axis.field: value})
        rows.extend(run(cfg, cache_dir=cache_dir))
    return SweepResult(axis=axis, base=base_config, rows=tuple(rows))
```

`SweepResult.to_plot(ax_metric)` returns a matplotlib figure with the right shape per axis type (bar for categorical, line for numeric). Plots are rendered into PNG + SVG and embedded in the notebooks; the underlying CSVs are also committed for downstream reuse.

## 7. Acceptance criteria

This sub-project is "done" when:

- All 7 extended datasets have a notebook that executes end-to-end on a fresh devcontainer.
- All 6 ablation axes have a notebook with a published plot and a CSV.
- All 4 scaling studies have a notebook with a published plot and a CSV.
- The `summary.ipynb` table compares `mononet` to XGBoost and unconstrained MLP across all 12 datasets (5 paper + 7 extended) on a unified metric scale.
- The "negative control" Higgs notebook shows `mononet` with an all-zero mask matches an unconstrained MLP within noise — confirming the construction does not pay an overhead when no monotonicity is declared.
- The library's worst-case relative parameter count and runtime overhead vs. unconstrained MLP at equal accuracy is documented (one bullet in the docs intro).

## 8. Open items

- **FICO HELOC redistribution.** The dataset requires registration; we likely cannot ship preprocessed data. Loader must fetch on first use with clear instructions. Confirm license terms permit benchmark publication.
- **Higgs subset size.** The full 11M rows are too large to ship; 1M-row subset is standard but check whether OpenML hosts a canonical "Higgs-1M". If not, downsample-and-cache.
- **Ablation seed budget.** 10 seeds × 6 axes × N levels = budget-sensitive. Default to **5 seeds** for ablation plots, **10 seeds** for headline tables, **3 seeds** for scaling studies — document the convention prominently in plots.
- **MovieLens-1M target.** Rating prediction (regression) or implicit recommendation (binary)? Lean regression for clarity of the monotonic prior.
- **Compute budget.** Approximate the total GPU-hours and decide which devcontainer flavor to standardize on for the canonical run.

## 9. What is intentionally NOT in this sub-project

- **Tables 1 / 2 reproduction.** Owned by Sub-project B.
- **New layer variants** — invertible / strictly-monotonic / per-feature-temperature. Sub-project D introduces strict monotonicity; we benchmark what already exists.
- **Production-grade tabular-deep-learning comparators** (TabNet, FT-Transformer, SAINT, etc.). The paper compares to *monotonic* baselines; this sub-project adds extended *monotonic* benchmarks. Comparing against general tabular DL is a different question and is out of scope.
- **Public datasets that contain PII without clear redistribution rights.** Even if the dataset is famous, if license is murky we skip.
