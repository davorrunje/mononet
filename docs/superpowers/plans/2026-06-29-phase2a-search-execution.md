# Phase-2a Search Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Phase-2a 20-study search runnable and reproducible via a Typer CLI over a DRY `run_dataset()`, fix/extend the execution tooling, and enhance the headline tables.

**Architecture:** Extract the per-dataset 4-flavor loop (now inline in the search notebooks) into `run_dataset()` in the engine; a Typer CLI and the (rewritten one-liner) notebooks both call it. Thread `n_jobs` into `search()`. Fix `execute-benchmarks.sh`, add a wrapper, enhance the summary notebook, and document a runbook. Real searches stay manual; CI runs synthetic unit tests; the agent does a CPU `--smoke` validation (uncommitted results).

**Tech Stack:** Optuna, Typer, the merged Phase-2a engine (`search`, `final_eval`, `flavor_name`, `StudyResult`), Phase-1 `run`/`aggregate`/`registry`, xgboost, pytest.

## Global Constraints

- Build on the merged engine (`benchmarks/_common/search.py`): `search(bundle, *, mode, residual, backend, n_trials=50, seed=0, epochs=50, metric=None, storage=None) -> StudyResult`; `final_eval(bundle, best_params, *, mode, residual, backend, metric=None, seeds=range(10), epochs=50, top_k=5) -> Aggregate`; `flavor_name(mode, residual)`; `StudyResult(dataset, flavor, best_params, best_value, n_trials)`. `registry.load(name, *, data_dir)`; `download.default_dest()`.
- The four flavors: `("switch",False), ("switch",True), ("absolute",False), ("absolute",True)`; flavor string via `flavor_name`.
- Per-dataset budget defaults (centralize, single source of truth): `auto`/`heart`/`compas` → `n_trials=50, final_seeds=range(10), final_top_k=5`; `loan`/`blog` → `n_trials=25, final_seeds=range(5), final_top_k=3`. `epochs=50` for all.
- Per-flavor result JSON record keys (unchanged from the notebooks): `dataset, flavor, best_params, val_best, test_metric, test_mean, test_std, n_seeds, n_selected`. Default out dir `benchmarks/results/phase2`.
- `n_jobs` (default 1) threads `run_dataset` → `search()` → `study.optimize(..., n_jobs=n_jobs)` (additive; default preserves behaviour).
- CLI is module-invoked `python -m benchmarks.search` (Typer) + a `tools/mononet-benchmark-search` wrapper — **NOT** a `[project.scripts]` console script (would ship `benchmarks/` in the wheel). `typer` added to the `bench` group (`==` pin; repo-only).
- No `mononet` package change; `benchmarks/` stays out of the wheel; `DatasetBundle` unchanged.
- Real searches are manual; CI runs only synthetic unit tests (`importorskip` optuna + backend). The agent's `--smoke` run does NOT commit result JSON.
- Branch: `feat/phase2a-search-cli` (already created, holds the spec). Never commit to `main`. Commits signed; end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer. Run from repo root `/Users/davor/Projects/PhD/mononet`.

---

### Task 1: `n_jobs` in `search()` + extract `run_dataset()`

**Files:**
- Modify: `benchmarks/_common/search.py` (add `n_jobs` to `search()`; add `_BUDGET`, `run_dataset()`)
- Test: `tests/benchmarks/test_run_dataset.py`

**Interfaces:**
- Consumes: `search`, `final_eval`, `flavor_name`, `registry.load`, `download.default_dest`.
- Produces: `run_dataset(dataset, *, backend="torch", flavors=ALL4, n_trials=None, epochs=50, n_jobs=1, final_seeds=None, final_top_k=None, data_dir=None, out_dir=None, storage_dir=None) -> list[Path]`; `_BUDGET: dict[str, tuple[int, range, int]]`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_run_dataset.py` (uses the committed `auto` fixtures from Phase 1; real tiny end-to-end):

```python
import json
from pathlib import Path

import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.search import run_dataset

FIXTURES = Path(__file__).parent / "fixtures"


def test_run_dataset_writes_one_json_per_flavor(tmp_path: Path) -> None:
    paths = run_dataset(
        "auto", backend="torch",
        flavors=(("switch", False), ("absolute", False)),
        n_trials=2, epochs=1, final_seeds=range(2), final_top_k=2,
        data_dir=FIXTURES, out_dir=tmp_path,
    )
    assert len(paths) == 2
    assert {p.name for p in paths} == {"auto-switch-plain.json", "auto-absolute-plain.json"}
    rec = json.loads(paths[0].read_text())
    assert rec["dataset"] == "auto"
    assert {"flavor", "best_params", "val_best", "test_metric", "test_mean",
            "test_std", "n_seeds", "n_selected"} <= set(rec)


def test_run_dataset_default_budget_from_table() -> None:
    from benchmarks._common.search import _BUDGET

    assert _BUDGET["auto"][0] == 50 and _BUDGET["loan"][0] == 25


def test_storage_uses_deterministic_study_name_so_it_resumes(tmp_path: Path) -> None:
    # With a fixed study_name + shared storage, a second search RESUMES the same
    # study (accumulating trials) rather than starting a new one. This is what
    # makes --storage-dir resumable and same-study multi-worker concurrency possible.
    import optuna

    from benchmarks._common.bundle import DatasetBundle
    from benchmarks._common.search import flavor_name, search
    from benchmarks.datasets.registry import load

    bundle = load("auto", data_dir=FIXTURES)
    storage = f"sqlite:///{tmp_path}/auto.db"
    search(bundle, mode="switch", residual=False, backend="torch",
           n_trials=2, epochs=1, storage=storage)
    search(bundle, mode="switch", residual=False, backend="torch",
           n_trials=2, epochs=1, storage=storage)
    study = optuna.load_study(
        study_name=f"auto-{flavor_name('switch', False)}", storage=storage
    )
    assert len(study.trials) == 4  # resumed, not restarted
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_run_dataset.py -q`
Expected: FAIL (`run_dataset` not defined).

- [ ] **Step 3: Implement** — in `benchmarks/_common/search.py`.

First, fix and extend `search()`:
- Add `n_jobs: int = 1` to its keyword params; change the optimize call to
  `study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)`.
- **Add a deterministic `study_name` to `create_study`** so `load_if_exists` genuinely
  resumes and multiple workers can share one study:
  `study_name=f"{bundle.name}-{flavor_name(mode, residual)}"`. (The merged code omits
  `study_name`, so Optuna assigns a random one — meaning `--storage-dir` never actually
  resumes and same-study multi-process concurrency is impossible. This one line fixes both.)

So the call becomes:
```python
study = optuna.create_study(
    study_name=f"{bundle.name}-{flavor_name(mode, residual)}",
    direction=direction,
    sampler=optuna.samplers.TPESampler(seed=seed),
    storage=storage,
    load_if_exists=storage is not None,
)
...
study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
```

Then add the budget table and `run_dataset` (place near the bottom, after `final_eval`):

```python
from pathlib import Path  # add to existing imports if absent

_ALL_FLAVORS: tuple[tuple[str, bool], ...] = (
    ("switch", False), ("switch", True), ("absolute", False), ("absolute", True),
)
# (n_trials, final_seeds, final_top_k) per dataset
_BUDGET: dict[str, tuple[int, range, int]] = {
    "auto": (50, range(10), 5),
    "heart": (50, range(10), 5),
    "compas": (50, range(10), 5),
    "loan": (25, range(5), 3),
    "blog": (25, range(5), 3),
}


def run_dataset(
    dataset: str,
    *,
    backend: str = "torch",
    flavors: tuple[tuple[str, bool], ...] = _ALL_FLAVORS,
    n_trials: int | None = None,
    epochs: int = 50,
    n_jobs: int = 1,
    final_seeds: "Iterable[int] | None" = None,
    final_top_k: int | None = None,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    storage_dir: Path | None = None,
) -> list[Path]:
    """Search + final_eval each flavor of one dataset; write per-flavor JSON.

    Budget falls back to the per-dataset `_BUDGET` defaults when not overridden.
    Returns the written JSON paths.
    """
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    b_trials, b_seeds, b_topk = _BUDGET.get(dataset, (50, range(10), 5))
    n_trials = b_trials if n_trials is None else n_trials
    final_seeds = b_seeds if final_seeds is None else final_seeds
    final_top_k = b_topk if final_top_k is None else final_top_k
    data_dir = data_dir or default_dest()
    out_dir = out_dir or (Path(__file__).resolve().parents[1] / "results" / "phase2")
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load(dataset, data_dir=data_dir)
    written: list[Path] = []
    for mode, residual in flavors:
        fname = flavor_name(mode, residual)
        storage = (
            f"sqlite:///{storage_dir}/{dataset}-{fname}.db" if storage_dir else None
        )
        study = search(
            bundle, mode=mode, residual=residual, backend=backend,
            n_trials=n_trials, epochs=epochs, n_jobs=n_jobs, storage=storage,
        )
        agg = final_eval(
            bundle, study.best_params, mode=mode, residual=residual, backend=backend,
            seeds=final_seeds, epochs=epochs, top_k=final_top_k,
        )
        rec = {
            "dataset": dataset, "flavor": study.flavor,
            "best_params": study.best_params, "val_best": study.best_value,
            "test_metric": agg.metric, "test_mean": agg.mean, "test_std": agg.std,
            "n_seeds": agg.n_seeds, "n_selected": agg.n_selected,
        }
        path = out_dir / f"{dataset}-{fname}.json"
        path.write_text(json.dumps(rec, indent=2))
        written.append(path)
    return written
```

(Add `import json` and `from pathlib import Path` to the module imports if not present;
keep `Iterable` import handling consistent with the file's `TYPE_CHECKING` style.)

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_run_dataset.py -q`
Expected: PASS (2 passed). Run `uv run ruff check benchmarks/ tests/benchmarks/` and the canonical `uv run mypy` — both clean. Also confirm the existing `tests/benchmarks/test_search.py` still passes (search() signature change is additive).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/search.py tests/benchmarks/test_run_dataset.py
git commit -m "$(cat <<'EOF'
bench: extract run_dataset() + thread n_jobs through search()

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Typer CLI + dep + wrapper

**Files:**
- Create: `benchmarks/search.py` (Typer app + `__main__`)
- Create: `tools/mononet-benchmark-search`
- Modify: `pyproject.toml` (`bench` group += typer)
- Test: `tests/benchmarks/test_search_cli.py`

**Interfaces:**
- Consumes: `run_dataset` (Task 1).
- Produces: a Typer `app`; `python -m benchmarks.search` entrypoint; a `_SMOKE` preset.

- [ ] **Step 1: Add typer to the bench group + lock**

In `pyproject.toml` `[dependency-groups].bench`, add a `typer` `==` pin (after `optuna`):

```toml
    "typer==0.20.0",
```

Run `uv lock` (use the resolved version if `0.20.0` doesn't resolve on 3.11–3.13; record it); `uv sync --group bench`; confirm `uv lock --check`.

- [ ] **Step 2: Write the failing test** — `tests/benchmarks/test_search_cli.py`:

```python
import pytest

pytest.importorskip("typer")
pytest.importorskip("optuna")

from typer.testing import CliRunner

from benchmarks.search import _SMOKE, app

runner = CliRunner()


def test_help_lists_flags() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for flag in ("--datasets", "--flavors", "--backend", "--n-trials", "--n-jobs", "--smoke", "--dry-run"):
        assert flag in res.output


def test_smoke_preset_values() -> None:
    assert _SMOKE["datasets"] == ["auto", "heart"]
    assert _SMOKE["n_trials"] == 5 and _SMOKE["epochs"] == 5


def test_dry_run_reports_plan_without_running() -> None:
    res = runner.invoke(app, ["--smoke", "--dry-run"])
    assert res.exit_code == 0
    assert "auto" in res.output and "heart" in res.output
    assert "would run" in res.output.lower()
```

- [ ] **Step 3: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search_cli.py -q`
Expected: FAIL (module missing).

- [ ] **Step 4: Implement** — `benchmarks/search.py`:

```python
"""Typer CLI to run the Phase-2a flavor search.

Invoke: `uv run python -m benchmarks.search [options]`
(or `tools/mononet-benchmark-search`). Repo-only; not a packaged console script.
"""

from __future__ import annotations

from pathlib import Path

import typer

from benchmarks._common.search import _ALL_FLAVORS, flavor_name, run_dataset

app = typer.Typer(add_completion=False, help="Run the Phase-2a HP-search flavor study.")

_ALL_DATASETS = ["auto", "heart", "compas", "loan", "blog"]
_SMOKE = {"datasets": ["auto", "heart"], "n_trials": 5, "epochs": 5,
          "final_seeds": 2, "final_top_k": 2}


def _parse_flavors(spec: str | None) -> tuple[tuple[str, bool], ...]:
    if not spec:
        return _ALL_FLAVORS
    out = []
    for name in spec.split(","):
        mode, _, kind = name.partition("-")
        out.append((mode, kind == "residual"))
    return tuple(out)


@app.command()
def main(  # noqa: PLR0913 - a CLI naturally has many flags
    datasets: str = typer.Option(",".join(_ALL_DATASETS), "--datasets"),
    flavors: str = typer.Option("", "--flavors", help="e.g. switch-plain,absolute-residual"),
    backend: str = typer.Option("torch", "--backend"),
    n_trials: int | None = typer.Option(None, "--n-trials"),
    epochs: int = typer.Option(50, "--epochs"),
    n_jobs: int = typer.Option(1, "--n-jobs"),
    final_seeds: int | None = typer.Option(None, "--final-seeds"),
    final_top_k: int | None = typer.Option(None, "--final-top-k"),
    out_dir: Path | None = typer.Option(None, "--out-dir"),
    storage_dir: Path | None = typer.Option(None, "--storage-dir"),
    smoke: bool = typer.Option(False, "--smoke", help="tiny preset for validation"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the plan, run nothing"),
) -> None:
    """Run search + final_eval for each (dataset, flavor) and write result JSON."""
    ds = _SMOKE["datasets"] if smoke else [d for d in datasets.split(",") if d]
    n_trials = _SMOKE["n_trials"] if smoke else n_trials
    epochs = _SMOKE["epochs"] if smoke else epochs
    fseeds = _SMOKE["final_seeds"] if smoke else final_seeds
    ftopk = _SMOKE["final_top_k"] if smoke else final_top_k
    flavs = _parse_flavors(flavors)
    flav_names = [flavor_name(m, r) for m, r in flavs]

    if dry_run:
        typer.echo(
            f"would run datasets={ds} flavors={flav_names} backend={backend} "
            f"n_trials={n_trials} epochs={epochs} n_jobs={n_jobs}"
        )
        raise typer.Exit(0)

    for dataset in ds:
        paths = run_dataset(
            dataset, backend=backend, flavors=flavs, n_trials=n_trials, epochs=epochs,
            n_jobs=n_jobs,
            final_seeds=range(fseeds) if fseeds is not None else None,
            final_top_k=ftopk, out_dir=out_dir, storage_dir=storage_dir,
        )
        typer.echo(f"{dataset}: wrote {len(paths)} result files")


if __name__ == "__main__":
    app()
```

`tools/mononet-benchmark-search` (make executable, `chmod +x`):

```bash
#!/usr/bin/env bash
# Wrapper for the repo-only Phase-2a search CLI (not a packaged console script).
exec uv run python -m benchmarks.search "$@"
```

- [ ] **Step 5: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search_cli.py -q`
Expected: PASS (3 passed). ruff + canonical mypy clean. Smoke-check the wrapper:
`chmod +x tools/mononet-benchmark-search && ./tools/mononet-benchmark-search --smoke --dry-run` prints the plan.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/search.py tools/mononet-benchmark-search pyproject.toml uv.lock tests/benchmarks/test_search_cli.py
git commit -m "$(cat <<'EOF'
bench: Typer search CLI (python -m benchmarks.search) + wrapper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Rewrite the 5 search notebooks to `run_dataset` one-liners

**Files:**
- Modify: `benchmarks/notebooks/search-{auto,heart,compas,loan,blog}.ipynb`

**Interfaces:**
- Consumes: `run_dataset` (Task 1).

- [ ] **Step 1: Replace each notebook's code cell** with a one-liner call (keep the markdown header). For `search-auto.ipynb` the code cell becomes:

```python
from benchmarks._common.search import run_dataset

# Per-dataset budget defaults live in benchmarks/_common/search.py (_BUDGET).
# Override here if exploring, e.g. run_dataset("auto", n_trials=20).
paths = run_dataset("auto", backend="torch", storage_dir="../results/phase2")
print("wrote:", [p.name for p in paths])
```

The other four are identical with the dataset name (`heart`, `compas`, `loan`, `blog`). The
per-dataset budget now comes from `_BUDGET` automatically, so the notebooks no longer carry
`N_TRIALS`/seeds knobs (the markdown header still documents the dataset's budget). Notebooks
remain scaffolds with NO executed outputs.

- [ ] **Step 2: Verify valid JSON + run_dataset reference**

Run:
```bash
uv run python -c "import json,glob; [json.load(open(f)) for f in glob.glob('benchmarks/notebooks/*.ipynb')]; print('valid JSON')"
grep -L "run_dataset" benchmarks/notebooks/search-*.ipynb || echo "all reference run_dataset"
```
Expected: `valid JSON`; every notebook references `run_dataset` (the `grep -L` prints nothing).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/notebooks/
git commit -m "$(cat <<'EOF'
bench: search notebooks call run_dataset (DRY with the CLI)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Fix `tools/execute-benchmarks.sh`

**Files:**
- Modify: `tools/execute-benchmarks.sh`

- [ ] **Step 1: Repoint the nbconvert glob.** Change the stale `docs/docs/benchmarks/*.ipynb` to the live tree and add a clarifying comment. The execute line becomes:

```bash
echo ">>> executing rendered benchmark notebooks under docs/benchmarks/"
# NOTE: the Phase-2a SEARCH notebooks (benchmarks/notebooks/) are NOT executed here —
# run them via the CLI: tools/mononet-benchmark-search (see benchmarks/README.md).
uv run jupyter nbconvert \
  --to notebook \
  --execute \
  --inplace \
  --ExecutePreprocessor.timeout=14400 \
  docs/benchmarks/**/*.ipynb docs/benchmarks/*.ipynb
```

(Confirm the shell has `globstar` or list the explicit subdirs — if `bash` without
`shopt -s globstar`, add `shopt -s globstar` near the top or enumerate
`docs/benchmarks/paper-reproduction/*.ipynb docs/benchmarks/*.ipynb`.)

- [ ] **Step 2: Verify the script is syntactically valid and globs resolve**

Run:
```bash
bash -n tools/execute-benchmarks.sh && echo "syntax ok"
ls docs/benchmarks/*.ipynb docs/benchmarks/paper-reproduction/*.ipynb | head
```
Expected: `syntax ok`; the globs list the real notebooks (do NOT actually execute them here).

- [ ] **Step 3: Commit**

```bash
git add tools/execute-benchmarks.sh
git commit -m "$(cat <<'EOF'
tools: fix execute-benchmarks.sh path (docs/docs/benchmarks -> docs/benchmarks)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Enhance `flavor-comparison.ipynb` (flavors vs paper-quoted vs XGBoost)

**Files:**
- Modify: `docs/benchmarks/flavor-comparison.ipynb`

**Interfaces:**
- Consumes: committed `benchmarks/results/phase2/*.json`; `benchmarks.baselines.xgboost.run_xgboost`; `registry.load`.

- [ ] **Step 1: Replace the code cell** so that, per dataset, it builds a table with one row per flavor (from the JSON), a `paper (CMNN)` row, and an `XGBoost` row. Keep the missing-results guard. The cell:

```python
import json
from pathlib import Path

import pandas as pd

from benchmarks.datasets.download import default_dest
from benchmarks.datasets.registry import DATASETS, load
from benchmarks.baselines.xgboost import run_xgboost

RESULTS = Path("../../benchmarks/results/phase2")

# Paper-quoted Table 1/2 numbers (tagged [quoted]). Blog is RMSE; others per dataset.
PAPER = {
    "auto": ("mse", 8.37),
    "heart": ("accuracy", 0.885),
    "compas": ("accuracy", 0.692),
    "loan": ("accuracy", 0.653),
    "blog": ("rmse", None),   # TODO(maintainer): transcribe paper Table 1 Blog RMSE
}

results = sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []
if not results:
    print("No Phase-2a results committed yet. Run tools/mononet-benchmark-search.")
else:
    by_ds: dict[str, list[dict]] = {}
    for f in results:
        rec = json.loads(f.read_text())
        by_ds.setdefault(rec["dataset"], []).append(rec)
    for ds, recs in sorted(by_ds.items()):
        metric = recs[0]["test_metric"]
        rows = []
        for rec in sorted(recs, key=lambda r: r["flavor"]):
            mean = rec["test_mean"]
            # Blog: study reports mse; show rmse for like-for-like paper comparison.
            if ds == "blog" and metric == "mse":
                mean = mean ** 0.5
            rows.append({"method": rec["flavor"], "value": round(mean, 4),
                         "std": round(rec["test_std"], 4)})
        pmetric, pval = PAPER.get(ds, (metric, None))
        rows.append({"method": "paper (CMNN) [quoted]", "value": pval, "std": "-"})
        try:
            xgb = run_xgboost(load(ds, data_dir=default_dest()), seed=0)
            key = "accuracy" if "accuracy" in xgb else ("rmse" if ds == "blog" else "mse")
            rows.append({"method": "XGBoost", "value": round(xgb[key], 4), "std": "-"})
        except Exception as exc:  # data not downloaded in the render env
            rows.append({"method": "XGBoost", "value": f"(skipped: {type(exc).__name__})", "std": "-"})
        print(f"### {ds}  (metric: {'rmse' if ds=='blog' else metric})")
        display(pd.DataFrame(rows).set_index("method"))
```

(Update the markdown cell to note the table shows the four flavors at their tuned best
alongside the paper-quoted number and XGBoost, and that Blog is shown as RMSE.)

- [ ] **Step 2: Verify valid JSON + strict docs build**

Run:
```bash
uv run python -c "import json; json.load(open('docs/benchmarks/flavor-comparison.ipynb')); print('valid JSON')"
./tools/build-docs.sh
```
Expected: `valid JSON`; docs build exit 0 (no results committed → the guard prints the
placeholder; build stays green).

- [ ] **Step 3: Commit**

```bash
git add docs/benchmarks/flavor-comparison.ipynb
git commit -m "$(cat <<'EOF'
docs: flavor-comparison shows flavors vs paper-quoted vs XGBoost

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Runbook in `benchmarks/README.md`

**Files:**
- Modify: `benchmarks/README.md`

- [ ] **Step 1: Replace/extend the Phase-2a section** with a "Running the search" runbook covering: compute (Apple-Silicon → CPU native/`default` devcontainer, CUDA host → `gpu-torch`; CPU is fine for these tiny nets); `python -m benchmarks.datasets.download`; `tools/mononet-benchmark-search` (full budget) writing `results/phase2/*.json` + git-ignored `*.db`; resumability via `--storage-dir` (Optuna SQLite + the deterministic `study_name` from Task 1 — re-running the same dataset/flavor resumes that study); reproducibility (fixed seed); the parallelism levers — process-level across **different** studies (`for d in ...; do tools/mononet-benchmark-search --datasets "$d" & done; wait`), in-study threaded `--n-jobs`, and **same-study multi-worker** (several processes sharing one `--storage-dir`+study_name fill `n_trials` collectively; SQLite suits a few workers, a server DB for heavy fan-out); re-render `docs/benchmarks/flavor-comparison.ipynb`; and **commit** the result JSON + re-rendered summary (not `*.db`/`*.jsonl`). Use the exact content from spec §6.

- [ ] **Step 2: Verify the README mentions the key commands**

Run:
```bash
grep -qE "mononet-benchmark-search" benchmarks/README.md && grep -qE "n-jobs|n_jobs" benchmarks/README.md && grep -qE "gpu-torch" benchmarks/README.md && echo "runbook keys present"
```
Expected: `runbook keys present`.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/README.md
git commit -m "$(cat <<'EOF'
docs: Phase-2a search runbook (compute, parallelism, commit/repro)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the executor

- **Verify against the real gates each task:** `uv run ruff check`, `uv run ruff format --check`, the **canonical** `uv run mypy` (scans `tests/` too), and — before finishing — `uv run pre-commit run --all-files --hook-stage manual` (codespell/end-of-file-fixer have bitten this repo when only `ruff` was run). New code/abbreviations may need a `.codespell-whitelist.txt` entry.
- **Backend gating:** sync the active backend (`uv sync --extra <backend> --group bench`) before backend-touching tests so they run, not skip.
- **The agent runs the `--smoke` validation after the tasks**, as a controller step (not a task): `uv run python -m benchmarks.search --smoke --out-dir <tmp> --storage-dir <tmp>` on this CPU machine (needs the Zenodo `auto`/`heart` CSVs, or point `--out-dir`/data at fixtures), confirm finite metrics + a rendered table, and **do not commit the result JSON**.
- `typer==0.20.0` is a guess — use whatever `uv lock` resolves; typer pulls click (the repo already overrides `click>=8.2.1` for semgrep — confirm no conflict).
- Do not commit Zenodo data, `dist/`, `*.db`, or `*.jsonl`.
