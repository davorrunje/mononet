# Phase 2a Search Execution — CLI, Runbook, and Headline Tables — Design

**Date:** 2026-06-29
**Status:** Approved
**Builds on:** [Phase 2a HP-search engine + flavor study](2026-06-28-phase2a-hp-search-and-flavor-study-design.md) (PR #55, merged) — `search()`, `final_eval()`, the 5 search notebooks, `flavor-comparison.ipynb`.

> Goal: make the real 20-study Optuna search runnable and reproducible, produce the
> headline flavor-comparison tables, and validate the whole pipeline end-to-end. This is
> execution tooling + a runbook, not new modelling. Phase 2b (physics/medicine datasets)
> is separate.

## 1. Goals & non-goals

### Goals
- A **Typer CLI** (`python -m benchmarks.search`) that orchestrates the 20 `(dataset,
  flavor)` studies with flags for datasets, flavors, backend, and budget.
- **DRY:** one engine function the CLI *and* the existing notebooks call — no duplicated
  per-dataset loop.
- Fix the **stale `tools/execute-benchmarks.sh`** (points at the dead `docs/docs/benchmarks/`
  path) and add a `tools/mononet-benchmark-search` wrapper for the nice command name.
- Enhance `flavor-comparison.ipynb` to render, per dataset, the **4 flavors vs the
  paper-quoted number vs XGBoost**.
- A **runbook** (in `benchmarks/README.md`) for running the full search on `gpu-torch`.
- An agent-run **CPU smoke run** validating the end-to-end pipeline (no committed result JSON).

### Non-goals
- Not the full headline numbers themselves — those are the maintainer's `gpu-torch` run.
- No packaged console script (`[project.scripts]`) — would ship `benchmarks/` in the wheel
  (forbidden). The CLI is module-invoked + a `tools/` wrapper.
- No `mononet` package change; `benchmarks/` stays out of the wheel.
- No cross-backend study here (headline is `torch`); no new search axes.

## 2. DRY engine function

Extract the per-dataset 4-flavor loop (currently inline in each search notebook) into
`benchmarks/_common/search.py`:

```python
def run_dataset(
    dataset: str,
    *,
    backend: str = "torch",
    flavors: tuple[tuple[str, bool], ...] = (
        ("switch", False), ("switch", True), ("absolute", False), ("absolute", True),
    ),
    n_trials: int = 50,
    epochs: int = 50,
    n_jobs: int = 1,
    final_seeds: Iterable[int] = range(10),
    final_top_k: int = 5,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    storage_dir: Path | None = None,
) -> list[Path]:
    """Run search + final_eval for each flavor of one dataset; write per-flavor JSON.

    Returns the written JSON paths. Loads the bundle via registry.load; uses each
    flavor's Optuna study (TPE) then final_eval; writes
    out_dir/<dataset>-<flavor>.json (default out_dir = benchmarks/results/phase2).
    """
```

Per-dataset budget defaults (Loan/Blog reduced) move into a single table consulted by both
the CLI and the notebooks. The 5 search notebooks are rewritten to a one-liner
`run_dataset("<dataset>", backend=BACKEND)` so notebook and CLI share one code path.

## 3. The Typer CLI

`benchmarks/search.py` — a Typer app, invoked `uv run python -m benchmarks.search`:

```
python -m benchmarks.search \
  [--datasets auto,heart,compas,loan,blog] \   # default: all five
  [--flavors switch-plain,switch-residual,absolute-plain,absolute-residual] \  # default: all
  [--backend torch] \
  [--n-trials N] [--epochs E] [--n-jobs J] [--final-seeds K] [--final-top-k T] \
  [--out-dir DIR] [--storage-dir DIR] \
  [--smoke]
```

`--n-jobs` (default 1) is threaded through `run_dataset` to `search()`, which passes it to
Optuna `study.optimize(..., n_jobs=n_jobs)` for in-study threaded trial parallelism. This
requires adding an additive `n_jobs: int = 1` parameter to the merged `search()` (default
preserves current behaviour). Note: torch CPU training is CPU-bound, so threaded `n_jobs`
gives partial speedup (torch releases the GIL during ops); **process-level parallelism —
launching one CLI invocation per dataset — is the higher-throughput lever on a many-core
box** and is documented in the runbook.

- Maps flags → `run_dataset(...)` per requested dataset.
- `--smoke` is a preset: `--datasets auto,heart --n-trials 5 --epochs 5 --final-seeds 2
  --final-top-k 2` (fast end-to-end validation).
- Without `--n-trials`/`--epochs`, each dataset uses its committed per-dataset default
  budget (the Loan/Blog reductions from Phase 2a).
- Prints a per-flavor progress line and the written JSON paths.
- `typer` is added to the `bench` dependency group (`==` pin; repo-only, never shipped).

`tools/mononet-benchmark-search` (executable):

```bash
#!/usr/bin/env bash
exec uv run python -m benchmarks.search "$@"
```

## 4. Fix `tools/execute-benchmarks.sh`

It currently `nbconvert`s `docs/docs/benchmarks/*.ipynb` (dead MkDocs path). Repoint it at
the current rendered benchmark notebooks under `docs/benchmarks/**/*.ipynb` (paper
reproduction + `flavor-comparison.ipynb`). It does **not** run the `benchmarks/notebooks/`
search notebooks (those are driven by the CLI/`run_dataset`); add a one-line comment saying
so. Keep the `airtai/monotonic-nn --no-deps` install step.

## 5. Headline tables (`flavor-comparison.ipynb`)

Enhance the rendered summary so, per dataset, it shows a row per **flavor** (test metric
mean ± std from the committed JSON), a **paper-quoted** row (the Table-1/2 numbers, a small
committed constant dict), and an **XGBoost** row (via the existing
`benchmarks.baselines.xgboost.run_xgboost` on the loaded bundle). Keep the
missing-results guard (renders a "no results yet" message when the JSON isn't present, so
the docs build stays green before the maintainer's run). Metric per dataset matches the
study metric (accuracy / mse; note rmse for Blog if the maintainer reports it).

## 6. Runbook (`benchmarks/README.md`)

A "Running the Phase-2a search" section:
1. **Compute.** The headline backend is `torch`. The models are tiny (≈2×21 MLPs on small
   tabular data), so the runner trains on **CPU** (no device handling; MPS/CUDA not used) —
   and CPU is the right device at this scale (MPS overhead doesn't pay off for tiny models).
   - **Apple Silicon (e.g. M-series MacBook):** run natively / in the `default` devcontainer
     on CPU. The full 20-study search is feasible on a modern Apple-Silicon CPU (`loan`/`blog`
     are the slow legs, at reduced budget). CUDA is unavailable on macOS — the `gpu-torch`
     devcontainer cannot run there.
   - **CUDA host:** use the `gpu-torch` devcontainer for faster `loan`/`blog`.
   - `--backend jax/keras` is the cross-backend option, not the headline.
2. `python -m benchmarks.datasets.download` to fetch the Zenodo CSVs.
3. `tools/mononet-benchmark-search` (full budget) — writes `results/phase2/*.json` and
   git-ignored `*.db` study files. Resumable: re-running with the same `--storage-dir`
   continues an interrupted study (Optuna SQLite). Reproducible: fixed Optuna `seed`.
4. Re-render the summary: `uv run jupyter nbconvert --to notebook --execute --inplace
   docs/benchmarks/flavor-comparison.ipynb`.
5. **Commit:** the `results/phase2/<dataset>-<flavor>.json` files and the re-rendered
   `flavor-comparison.ipynb` (with outputs). Do **not** commit `*.db`/`*.jsonl`.
6. **Parallelism (many-core hosts, e.g. 32-core ThreadRipper).** Two levers:
   - *Process-level (preferred for CPU-bound training):* run one CLI invocation per dataset
     concurrently — each writes its own JSON, no contention:
     ```bash
     for d in auto heart compas loan blog; do tools/mononet-benchmark-search --datasets "$d" & done; wait
     ```
   - *In-study:* `--n-jobs J` parallelizes trials within a study (threaded; partial speedup).
   These compose (e.g. 5 dataset processes × a few `--n-jobs` each), but keep the total under
   the core count. GPUs add little for these tiny nets; a `jax`/`keras` cross-backend run on a
   CUDA host auto-uses the GPU if desired.
7. Rough runtime guidance: auto/heart/compas small (minutes/flavor); loan/blog larger
   (reduced budget) — document approximate wall-clock after the first real run.

## 7. The smoke run (agent, now, CPU)

Run `python -m benchmarks.search --smoke` on this CPU machine to validate end-to-end:
search → per-flavor JSON written → `flavor-comparison.ipynb` renders a populated table.
Confirm finite metrics and a rendered table. **Do not commit the smoke result JSON** (it
would masquerade as real headline numbers); the committed summary notebook stays in its
"renders when results present" state. The smoke evidence (command + output) lives in the
task report only.

## 8. Testing / CI

- A fast unit test for the CLI/`run_dataset` wiring on a **synthetic** bundle (2 flavors, 2
  trials, 1 epoch) asserting it writes the expected JSON files and returns their paths
  (`pytest.importorskip` optuna + backend). No real datasets, no network in CI.
- A test that `--smoke` flag maps to the documented small preset.
- ruff + mypy clean (incl. the Typer app); strict docs build green (the enhanced summary
  notebook tolerates missing results).
- Real searches remain manual; CI never runs them.

## 9. Acceptance

- `python -m benchmarks.search --help` works; `--smoke` and full invocations run
  `run_dataset` and write `results/phase2/*.json`.
- The 5 search notebooks call `run_dataset` (no duplicated loop); behaviour unchanged.
- `tools/execute-benchmarks.sh` targets the live `docs/benchmarks/` notebooks; the wrapper
  script works.
- `flavor-comparison.ipynb` renders flavors + paper-quoted + XGBoost when JSON is present,
  and a clean placeholder when absent (docs build green).
- CI unit tests pass (synthetic); ruff/mypy/pre-commit/static-analysis clean; no `mononet`
  change; `benchmarks/` not in the wheel.
- Agent smoke run validated end-to-end (reported, not committed as results).

## 10. Open items

- Per-dataset full-budget wall-clock is unknown until the first `gpu-torch` run; the runbook
  ships with estimates and a note to refine them.
- Paper-quoted constants: transcribe Table 1/2 values (COMPAS acc, Blog RMSE, Loan acc,
  AutoMPG MSE, Heart acc) into the summary notebook's reference dict, tagged `[quoted]`.
- Blog metric: studies optimize/report `mse` (Phase-2a default); the paper reports RMSE —
  the summary should display `rmse = sqrt(mse)` for Blog to compare like-for-like, or note
  the unit difference.
