# benchmarks

Repo-only harness for reproducing paper experiments. **Not shipped** in the `mononet` wheel or sdist.

## Benchmark protocol

Standard held-out protocol: fixed published train/test split; HP selection by
k-fold CV on **train** (5 folds for auto/heart/compas, single 80/20 holdout for
loan/blog); refit the selected config on full train; report **mean ± std over all
seeds** on the held-out test set (no best-k). The test set never influences model
selection.

Our numbers therefore sit somewhat higher than the originally published figures,
which used a test-selected protocol (validation-on-test, early stopping on test,
best-epoch, best-5-of-10) and are **not directly comparable**. Full explanation:
[docs/benchmarks/protocol.md](../docs/benchmarks/protocol.md).

## Phase 2a: Hyperparameter Search

The five search notebooks run Optuna TPE over a configurable hyperparameter space on validation splits, then refit best parameters on the full training set for final test evaluation.

### Setup

```bash
uv sync --group bench  # installs optuna and other benchmarking extras
```

### Running the Phase-2a Search

#### Compute Setup

The models are tiny (≈2×21 MLPs on small tabular data) and train on **CPU** — no device handling or GPU transfer overhead needed at this scale.

- **Apple Silicon (M-series MacBook):** Run natively or in the `default` devcontainer on CPU. The full 20-study search is feasible on modern Apple-Silicon CPUs; `loan`/`blog` are the slower legs (reduced budget). CUDA is unavailable on macOS, so the `gpu-torch` devcontainer cannot run there.
- **CUDA host:** Use the `gpu-torch` devcontainer for faster `loan`/`blog` runs (if GPU overhead is a concern, CPU is still the default and is often faster for these tiny models).
- `--backend jax` or `--backend keras` enables cross-backend validation; the headline backend is `torch`.

#### Steps

1. **Download datasets:**
   ```bash
   python -m benchmarks.datasets.download
   ```
   Fetches the Zenodo CSVs (auto, heart, compas, loan, blog).

2. **Run the full search:**
   ```bash
   tools/mononet-benchmark-search
   ```
   This runs all 5 datasets × 4 flavors (20 studies total), writing:
   - `results/phase2/<dataset>-<flavor>.json` (committed; contains best params, validation best, test mean/std)
   - `results/phase2/<dataset>-<flavor>.db` (git-ignored; Optuna SQLite study database)

   Budget defaults (per dataset):
   - **auto**, **heart**, **compas**: 50 trials, 10 final-eval seeds (best-5-of-10)
   - **loan**, **blog**: 25 trials, 5 final-eval seeds (best-3-of-5) — reduced HP search footprint for constrained datasets

3. **Resumability:** If interrupted, re-run the same command with the same `--storage-dir` to resume:
   ```bash
   tools/mononet-benchmark-search --storage-dir /path/to/storage
   ```
   Optuna SQLite persists the study state and deterministic study name, so trials resume where they left off.

4. **Reproducibility:** The search uses a fixed Optuna seed, so re-runs with the same parameters produce identical trial sequences.

5. **Re-render the summary:**
   ```bash
   uv run jupyter nbconvert --to notebook --execute --inplace docs/benchmarks/flavor-comparison.ipynb
   ```
   This generates the headline flavor-comparison table (per dataset: 4 flavors vs paper-quoted numbers vs XGBoost).

6. **Commit:**
   ```bash
   git add results/phase2/*.json docs/benchmarks/flavor-comparison.ipynb
   git commit -m "results: Phase-2a search results and summary"
   ```
   Commit the `.json` results and re-rendered summary notebook. **Do not commit `*.db` or `*.jsonl` files** (they are git-ignored and intended for local resumability only).

#### Parallelism (Many-Core Hosts, e.g., 32-core ThreadRipper)

You have two independent parallelism levers; compose them but keep total worker count under your core count:

1. **Process-level (preferred for CPU-bound training):** Launch one CLI invocation per dataset in the background, each writing its own JSON with no contention:
   ```bash
   for d in auto heart compas loan blog; do tools/mononet-benchmark-search --datasets "$d" & done; wait
   ```
   This is the highest-throughput lever on many-core boxes because torch CPU training is CPU-bound and releases the GIL during operations.

2. **In-study threaded parallelism:** Use `--n-jobs J` to parallelize trials within a single study (threaded; provides partial speedup):
   ```bash
   tools/mononet-benchmark-search --datasets auto --n-jobs 4
   ```

3. **Multi-worker same-study:** Several processes can share one `--storage-dir` and study name to fill `n_trials` collectively (Optuna SQLite coordinates them). SQLite suits a few workers; for heavy fan-out, use a server database (PostgreSQL, etc.). Example:
   ```bash
   # In process 1: tools/mononet-benchmark-search --datasets auto --storage-dir /shared/path &
   # In process 2: tools/mononet-benchmark-search --datasets auto --storage-dir /shared/path &
   # Both fill the same study (same dataset, flavor, backend).
   ```

Example: on a 32-core ThreadRipper, run 5 dataset processes with `--n-jobs 6` each (30 threads total), leaving 2 cores for the OS:
```bash
for d in auto heart compas loan blog; do
  tools/mononet-benchmark-search --datasets "$d" --n-jobs 6 &
done
wait
```

#### Approximate Wall-Clock Runtime

(To be updated after the first full `gpu-torch` run; CPU times are fast, GPU benefit for these tiny nets is marginal.)

### Matrix

20 studies total: 5 datasets × 4 flavors. All `.json` results are committed.
