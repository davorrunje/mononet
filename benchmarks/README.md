# benchmarks

Repo-only harness for reproducing paper experiments. **Not shipped** in the `mononet` wheel or sdist.

## Phase 2a: Hyperparameter Search

The five search notebooks run Optuna TPE over a configurable hyperparameter space on validation splits, then refit best parameters on the full training set for final test evaluation.

### Setup

```bash
uv sync --group bench  # installs optuna and other benchmarking extras
```

### Running a Search Notebook

Navigate to a `gpu-*` devcontainer (CPU variants lack CUDA):

```bash
# In .devcontainer/gpu-torch, gpu-jax, or gpu-keras
cd benchmarks
jupyter notebook notebooks/search-<dataset>.ipynb
```

Each notebook (`auto`, `heart`, `compas`, `loan`, `blog`):
- Loads the dataset and defines monotone features
- Runs 4 flavor configurations: `(switch, plain)`, `(switch, residual)`, `(absolute, plain)`, `(absolute, residual)`
- Per flavor: 50 trials (25 for `loan`/`blog` — smaller budgets) over validation, then final-eval on test with best-5-of-10 seeds (3-of-5 for `loan`/`blog`)
- Writes `results/phase2/<dataset>-<flavor>.json` (committed; contains best params, validation best, test mean/std)
- Creates `results/phase2/<dataset>-<flavor>.db` (git-ignored; Optuna SQLite study database)

### Matrix

20 studies total: 5 notebooks × 4 flavors. Commit all `.json` results.

### Per-Dataset Budget Notes

- **auto**, **heart**, **compas**: 50 trials, 10 final-eval seeds (best-5-of-10)
- **loan**, **blog**: 25 trials, 5 final-eval seeds (best-3-of-5) — smaller HP search footprint for constrained datasets
