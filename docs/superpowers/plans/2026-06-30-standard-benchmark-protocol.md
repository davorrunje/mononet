# Standard Benchmark Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the inherited test-selected reporting (best-5-of-10, single noisy holdout) with the standard tabular-DL protocol — k-fold CV for HP selection, mean±std over all seeds on a fixed held-out test set — and document why our numbers differ from the prior papers.

**Architecture:** A new index-based `cv_splits()` helper drives the Optuna objective (per-trial = mean metric across folds); `final_eval()` reports all seeds; `run_dataset()`/CLI thread an `n_splits` parameter with per-dataset defaults; docs gain a `protocol.md` explaining the divergence. The shipped `mononet` package is untouched; all changes live under `benchmarks/` and `docs/`.

**Tech Stack:** Python 3.11, numpy, scikit-learn (`KFold`/`StratifiedKFold`/`train_test_split`), Optuna, Typer, pytest, Sphinx + myst-nb.

## Global Constraints

- **Branch:** all work on `feat/auto-headline-run` (extends open PR #62). Never commit to `main`.
- **Signing:** commit **unsigned** during subagent execution (`git -c commit.gpgsign=false commit …`); the controller re-signs the whole branch before push (Secretive Touch ID).
- **No `mononet` package change**; `benchmarks/` never enters the wheel; no `[project.scripts]`.
- **No Pydantic**; stdlib dataclasses only. MyST field-list docstrings on public functions (`:param:`/`:returns:`, no `:type:`). Types on every function.
- **ruff** line length 88; **strict mypy** must be run as `uv run --group bench mypy` (benchmarks/ imports typer/optuna). `uv run pre-commit run --all-files --hook-stage manual` must pass (codespell, end-of-file-fixer).
- **Result JSON** written with a trailing newline (`json.dumps(...) + "\n"`). Never commit `*.db`/`*.jsonl`.
- **Preserve** lazy backend imports and the deterministic Optuna `study_name`.
- `n_splits` semantics, fixed verbatim: **≥2** ⇒ (stratified for `binary_classification`, else plain) K-fold; **==1** ⇒ single 80/20 holdout (same frac/stratify default as `train_val_split`).
- Per-dataset `_BUDGET` defaults: `n_splits` = **5** for auto/heart/compas, **1** for loan/blog. Seeds default **10**; loan/blog keep their reduced `range(5)`.
- Out of scope: Friedman/Nemenyi stats; re-running the other 4 datasets; the two-branch-vs-unified-`MonoDense` architecture divergence.

---

### Task 1: `cv_splits()` index helper

**Files:**
- Modify: `benchmarks/_common/splits.py` (add `cv_splits`; keep `train_val_split`)
- Test: `tests/benchmarks/test_splits.py`

**Interfaces:**
- Consumes: `DatasetBundle` (`.X_train`, `.y_train`, `.task`).
- Produces: `cv_splits(bundle, *, n_splits=5, seed, stratify=None) -> list[tuple[np.ndarray, np.ndarray]]` — a list of `(train_idx, val_idx)` integer-index arrays over the train rows. `n_splits>=2` → that many folds; `n_splits==1` → one 80/20 holdout. Never reads test.

- [ ] **Step 1: Write the failing tests**

Append to `tests/benchmarks/test_splits.py`:

```python
from benchmarks._common.splits import cv_splits


def test_cv_splits_folds_partition_train_once() -> None:
    b = _bundle("regression", n=200)
    folds = cv_splits(b, n_splits=5, seed=0)
    assert len(folds) == 5
    val_sizes = [len(val) for _, val in folds]
    assert sum(val_sizes) == len(b.X_train)  # every row validated exactly once
    all_val = np.concatenate([val for _, val in folds])
    assert sorted(all_val.tolist()) == list(range(len(b.X_train)))
    for tr, val in folds:  # train/val disjoint, cover all rows
        assert set(tr.tolist()).isdisjoint(val.tolist())
        assert len(tr) + len(val) == len(b.X_train)


def test_cv_splits_single_is_holdout() -> None:
    b = _bundle("regression", n=200)
    folds = cv_splits(b, n_splits=1, seed=0)
    assert len(folds) == 1
    tr, val = folds[0]
    assert len(val) == 40  # 20% of 200
    assert len(tr) == 160


def test_cv_splits_stratified_for_binary() -> None:
    b = _bundle("binary_classification", n=200)
    for _, val in cv_splits(b, n_splits=5, seed=0):
        assert set(np.unique(b.y_train[val])) == {0.0, 1.0}


def test_cv_splits_deterministic_for_seed() -> None:
    b = _bundle("regression", n=200)
    a = cv_splits(b, n_splits=5, seed=3)
    c = cv_splits(b, n_splits=5, seed=3)
    for (a_tr, a_val), (c_tr, c_val) in zip(a, c, strict=True):
        assert np.array_equal(a_tr, c_tr)
        assert np.array_equal(a_val, c_val)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/benchmarks/test_splits.py -q`
Expected: FAIL — `ImportError: cannot import name 'cv_splits'`.

- [ ] **Step 3: Implement `cv_splits`**

In `benchmarks/_common/splits.py`, change the import line and add the function. Replace the top import block:

```python
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
```

and append after `train_val_split`:

```python
def cv_splits(
    bundle: DatasetBundle,
    *,
    n_splits: int = 5,
    seed: int,
    stratify: bool | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Cross-validation index folds over `bundle`'s train arrays.

    :param n_splits: number of folds; `>= 2` uses K-fold, `== 1` returns a single
        80/20 holdout (same fraction/stratify default as `train_val_split`).
    :param seed: deterministic shuffling/splitting seed.
    :param stratify: stratify on `y`; defaults to True for binary classification.
    :returns: list of `(train_idx, val_idx)` integer-index arrays into the train
        rows. `bundle.X_test`/`y_test` are never read.
    :raises ValueError: if `n_splits < 1`.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if stratify is None:
        stratify = bundle.task == "binary_classification"
    n = len(bundle.X_train)
    idx = np.arange(n)
    if n_splits == 1:
        strat = bundle.y_train if stratify else None
        tr, val = train_test_split(
            idx, test_size=0.2, random_state=seed, stratify=strat
        )
        return [(tr, val)]
    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    return [(tr, val) for tr, val in splitter.split(idx, bundle.y_train)]
```

Also add `import numpy as np` at module top (it is currently only imported under `TYPE_CHECKING`). Replace:

```python
from typing import TYPE_CHECKING

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

if TYPE_CHECKING:
    import numpy as np

    from benchmarks._common.bundle import DatasetBundle
```

with:

```python
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/benchmarks/test_splits.py -q`
Expected: PASS (all split tests, old and new).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check benchmarks/_common/splits.py tests/benchmarks/test_splits.py && uv run --group bench mypy`
Expected: clean; `Success: no issues found`.

- [ ] **Step 6: Commit (unsigned during execution)**

```bash
git add benchmarks/_common/splits.py tests/benchmarks/test_splits.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): add cv_splits index helper (k-fold / single holdout)"
```

---

### Task 2: CV-driven `search()` objective

**Files:**
- Modify: `benchmarks/_common/search.py` (`search`, replace `_val_bundle` with `_fold_bundles`)
- Test: `tests/benchmarks/test_search.py`

**Interfaces:**
- Consumes: `cv_splits(...)` from Task 1; existing `run`, `suggest_config`, `aggregate`.
- Produces: `search(bundle, *, mode, residual, backend, n_trials=50, seed=0, epochs=50, n_jobs=1, n_splits=5, metric=None, storage=None) -> StudyResult` where `StudyResult.best_value` is the **mean CV metric** of the best trial.

- [ ] **Step 1: Write the failing test**

In `tests/benchmarks/test_search.py`, replace `test_search_two_trials_returns_finite_best` with a CV-aware version and add a folds test:

```python
def test_search_two_trials_two_folds_returns_finite_best() -> None:
    res = search(
        _bundle(),
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
        n_splits=2,
    )
    assert isinstance(res, StudyResult)
    assert res.n_trials == 2
    assert res.flavor == "switch-plain"
    assert np.isfinite(res.best_value)
    assert "lr" in res.best_params
    assert "width" in res.best_params


def test_search_objective_is_fold_mean() -> None:
    # n_splits=1 (single holdout) and n_splits=3 must both yield a finite CV metric;
    # this exercises the averaging path without asserting an exact value.
    for n_splits in (1, 3):
        res = search(
            _bundle(),
            mode="switch",
            residual=False,
            backend="torch",
            n_trials=2,
            seed=0,
            epochs=1,
            n_splits=n_splits,
        )
        assert np.isfinite(res.best_value)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_search.py -q`
Expected: FAIL — `search()` has no `n_splits` keyword (`TypeError: unexpected keyword argument 'n_splits'`).

- [ ] **Step 3: Implement the CV objective**

In `benchmarks/_common/search.py`:

Add `import numpy as np` after `import json` (top of file). Change the splits import:

```python
from benchmarks._common.splits import cv_splits
```

Replace `_val_bundle` with `_fold_bundles`:

```python
def _fold_bundles(
    bundle: DatasetBundle, *, n_splits: int, seed: int
) -> list[DatasetBundle]:
    """Throwaway per-fold bundles with each fold's validation rows in the test slot.

    Lets the search reuse run() (which evaluates on X_test) to score on every CV
    fold without ever touching the real test set.
    """
    folds = cv_splits(bundle, n_splits=n_splits, seed=seed)
    out: list[DatasetBundle] = []
    for tr, val in folds:
        out.append(
            dataclasses.replace(
                bundle,
                X_train=bundle.X_train[tr],
                y_train=bundle.y_train[tr],
                X_test=bundle.X_train[val],
                y_test=bundle.y_train[val],
            )
        )
    return out
```

Update `search` — add `n_splits: int = 5` to the signature (after `n_jobs: int = 1`) and average over folds:

```python
def search(
    bundle: DatasetBundle,
    *,
    mode: str,
    residual: bool,
    backend: str,
    n_trials: int = 50,
    seed: int = 0,
    epochs: int = 50,
    n_jobs: int = 1,
    n_splits: int = 5,
    metric: str | None = None,
    storage: str | None = None,
) -> StudyResult:
    """Tune (dataset, flavor) HPs by mean k-fold CV metric via Optuna TPE."""
    metric = metric or _primary_metric(bundle)
    direction = "minimize" if _lower_is_better(metric) else "maximize"
    folds = _fold_bundles(bundle, n_splits=n_splits, seed=seed)

    def objective(trial: optuna.Trial) -> float:
        cfg: BenchmarkConfig = suggest_config(
            trial,
            dataset=bundle.name,
            backend=backend,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            residual=residual,
            epochs=epochs,  # type: ignore[arg-type]
            metric=metric,  # type: ignore[arg-type]
        )
        scores: list[float] = []
        for fb in folds:
            rows = run(cfg, fb)
            if not rows:
                raise RuntimeError("run() returned no rows for trial")
            scores.append(float(rows[0].scores[metric]))  # type: ignore[index]
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=f"{bundle.name}-{flavor_name(mode, residual)}",
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return StudyResult(
        dataset=bundle.name,
        flavor=flavor_name(mode, residual),
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/benchmarks/test_search.py -q`
Expected: PASS for the two new tests (and `test_flavor_name`). `test_final_eval_returns_aggregate_on_test` is updated in Task 3.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check benchmarks/_common/search.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add benchmarks/_common/search.py tests/benchmarks/test_search.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): search objective = mean k-fold CV metric"
```

---

### Task 3: `final_eval()` reports all seeds (drop best-k)

**Files:**
- Modify: `benchmarks/_common/search.py` (`final_eval`)
- Test: `tests/benchmarks/test_search.py`

**Interfaces:**
- Consumes: existing `aggregate(rows, *, metric, lower_is_better, top_k)` from `results.py` (unchanged).
- Produces: `final_eval(bundle, best_params, *, mode, residual, backend, metric=None, seeds=range(10), epochs=50) -> Aggregate` — **no `top_k`**; mean±std over all seeds, so `Aggregate.n_selected == Aggregate.n_seeds`.

- [ ] **Step 1: Update the test to drop `top_k` and assert all-seeds**

In `tests/benchmarks/test_search.py`, replace `test_final_eval_returns_aggregate_on_test`:

```python
def test_final_eval_reports_all_seeds() -> None:
    b = _bundle()
    res = search(
        b,
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        n_splits=2,
    )
    # 6 seeds > the old top_k=5 default, so all-seeds reporting is observable:
    # the old best-5-of-6 would give n_selected == 5; the new behaviour gives 6.
    agg = final_eval(
        b,
        res.best_params,
        mode="switch",
        residual=False,
        backend="torch",
        seeds=range(6),
        epochs=1,
    )
    assert np.isfinite(agg.mean)
    assert agg.n_seeds == 6
    assert agg.n_selected == 6  # all seeds reported, no best-k selection
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_search.py::test_final_eval_reports_all_seeds -q`
Expected: FAIL — `final_eval()` still accepts/uses `top_k=5`, so `n_selected == 5 != 6`. (The call also no longer passes `top_k`, which is the signature change Step 3 makes.)

- [ ] **Step 3: Implement — remove `top_k`, aggregate over all rows**

In `benchmarks/_common/search.py`, change `final_eval`'s signature (remove `top_k: int = 5`) and its `aggregate` call:

Signature line becomes:

```python
def final_eval(
    bundle: DatasetBundle,
    best_params: dict[str, Any],
    *,
    mode: str,
    residual: bool,
    backend: str,
    metric: str | None = None,
    seeds: Iterable[int] = range(10),
    epochs: int = 50,
) -> Aggregate:
    """Refit best HPs on the full train split; report TEST mean±std over all seeds."""
```

And the final two statements become:

```python
    rows = run(cfg, bundle)
    return aggregate(
        rows, metric=metric, lower_is_better=_lower_is_better(metric), top_k=len(rows)
    )
```

(`top_k=len(rows)` selects every seed → `n_selected == n_seeds`; `results.py` is unchanged.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/benchmarks/test_search.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check benchmarks/_common/search.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add benchmarks/_common/search.py tests/benchmarks/test_search.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): final_eval reports mean±std over all seeds (no best-k)"
```

---

### Task 4: `run_dataset()` budget + JSON schema

**Files:**
- Modify: `benchmarks/_common/search.py` (`_BUDGET`, `run_dataset`)
- Test: `tests/benchmarks/test_run_dataset.py`

**Interfaces:**
- Consumes: `search(..., n_splits=...)` (Task 2), `final_eval(...)` without `top_k` (Task 3).
- Produces: `run_dataset(dataset, *, backend="torch", flavors=_ALL_FLAVORS, n_trials=None, epochs=50, n_jobs=1, final_seeds=None, n_splits=None, data_dir=None, out_dir=None, storage_dir=None) -> list[Path]`. `_BUDGET[ds] = (n_trials, seeds_range, n_splits)`. Result JSON keys: `dataset, flavor, best_params, cv_best, test_metric, test_mean, test_std, n_seeds` (no `n_selected`).

- [ ] **Step 1: Update tests for new signature + JSON schema**

In `tests/benchmarks/test_run_dataset.py`, replace `test_run_dataset_writes_one_json_per_flavor` and `test_run_dataset_default_budget_from_table`:

```python
def test_run_dataset_writes_one_json_per_flavor(tmp_path: Path) -> None:
    paths = run_dataset(
        "auto",
        backend="torch",
        flavors=(("switch", False), ("absolute", False)),
        n_trials=2,
        epochs=1,
        final_seeds=range(2),
        n_splits=2,
        data_dir=FIXTURES,
        out_dir=tmp_path,
    )
    assert len(paths) == 2
    assert {p.name for p in paths} == {
        "auto-switch-plain.json",
        "auto-absolute-plain.json",
    }
    rec = json.loads(paths[0].read_text())
    assert rec["dataset"] == "auto"
    assert {
        "flavor",
        "best_params",
        "cv_best",
        "test_metric",
        "test_mean",
        "test_std",
        "n_seeds",
    } <= set(rec)
    assert "n_selected" not in rec
    assert "val_best" not in rec


def test_run_dataset_default_budget_from_table() -> None:
    from benchmarks._common.search import _BUDGET

    assert _BUDGET["auto"] == (50, range(10), 5)
    assert _BUDGET["loan"] == (25, range(5), 1)
    assert _BUDGET["blog"][2] == 1  # large datasets use single holdout
```

In `test_storage_uses_deterministic_study_name_so_it_resumes`, add `n_splits=2` to both `search(...)` calls (keeps the test fast; trial-count assertion is unchanged):

```python
    search(
        bundle,
        mode="switch",
        residual=False,
        backend="torch",
        n_trials=2,
        epochs=1,
        n_splits=2,
        storage=storage,
    )
```
(apply to both calls).

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_run_dataset.py -q`
Expected: FAIL — `run_dataset` still takes `final_top_k`; `_BUDGET` tuples still end in top-k; JSON still has `val_best`/`n_selected`.

- [ ] **Step 3: Implement the budget + JSON changes**

In `benchmarks/_common/search.py`, replace the `_BUDGET` block:

```python
# (n_trials, final_seeds, n_splits) per dataset.
# n_splits: 5-fold CV for small/medium datasets; 1 (single holdout) for the large
# ones (loan/blog), where a single split is already low-variance and 5x cheaper.
_BUDGET: dict[str, tuple[int, range, int]] = {
    "auto": (50, range(10), 5),
    "heart": (50, range(10), 5),
    "compas": (50, range(10), 5),
    "loan": (25, range(5), 1),
    "blog": (25, range(5), 1),
}
```

Replace `run_dataset`'s signature line `final_top_k: int | None = None,` with `n_splits: int | None = None,`, and replace the budget-unpacking + `search`/`final_eval`/`rec` section:

```python
    b_trials, b_seeds, b_splits = _BUDGET.get(dataset, (50, range(10), 5))
    n_trials = b_trials if n_trials is None else n_trials
    final_seeds = b_seeds if final_seeds is None else final_seeds
    n_splits = b_splits if n_splits is None else n_splits
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
            bundle,
            mode=mode,
            residual=residual,
            backend=backend,
            n_trials=n_trials,
            epochs=epochs,
            n_jobs=n_jobs,
            n_splits=n_splits,
            storage=storage,
        )
        agg = final_eval(
            bundle,
            study.best_params,
            mode=mode,
            residual=residual,
            backend=backend,
            seeds=final_seeds,
            epochs=epochs,
        )
        rec = {
            "dataset": dataset,
            "flavor": study.flavor,
            "best_params": study.best_params,
            "cv_best": study.best_value,
            "test_metric": agg.metric,
            "test_mean": agg.mean,
            "test_std": agg.std,
            "n_seeds": agg.n_seeds,
        }
        path = out_dir / f"{dataset}-{fname}.json"
        path.write_text(json.dumps(rec, indent=2) + "\n")
        written.append(path)
    return written
```

Also update the `run_dataset` docstring's first line to mention CV is unchanged in spirit (optional). Ensure the `final_top_k` name appears nowhere in the file (`grep -n final_top_k benchmarks/_common/search.py` returns nothing).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/benchmarks/test_run_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check benchmarks/_common/search.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add benchmarks/_common/search.py tests/benchmarks/test_run_dataset.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): per-dataset n_splits budget; JSON cv_best, drop n_selected"
```

---

### Task 5: CLI `--cv-folds` (drop `--final-top-k`)

**Files:**
- Modify: `benchmarks/search.py` (`_SMOKE`, `main`)
- Test: `tests/benchmarks/test_search_cli.py`

**Interfaces:**
- Consumes: `run_dataset(..., n_splits=...)` (Task 4).
- Produces: CLI flag `--cv-folds` (int, default per-dataset budget); `--final-top-k` removed; `_SMOKE` has `"cv_folds": 2` (no `final_top_k`).

- [ ] **Step 1: Update the CLI tests**

In `tests/benchmarks/test_search_cli.py`: add `"--cv-folds"` to the flag list in `test_help_lists_flags`, and update `test_smoke_preset_values`:

```python
def test_smoke_preset_values() -> None:
    assert _SMOKE["datasets"] == ["auto", "heart"]
    assert _SMOKE["n_trials"] == 5
    assert _SMOKE["epochs"] == 5
    assert _SMOKE["cv_folds"] == 2
    assert "final_top_k" not in _SMOKE
```

And in `test_help_lists_flags` the loop tuple becomes:

```python
    for flag in (
        "--datasets",
        "--flavors",
        "--backend",
        "--n-trials",
        "--n-jobs",
        "--cv-folds",
        "--smoke",
        "--dry-run",
    ):
        assert flag in output
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_search_cli.py -q`
Expected: FAIL — `_SMOKE["cv_folds"]` KeyError / `--cv-folds` not in help.

- [ ] **Step 3: Implement the CLI change**

In `benchmarks/search.py`, replace `_SMOKE`:

```python
_SMOKE: dict[str, Any] = {
    "datasets": ["auto", "heart"],
    "n_trials": 5,
    "epochs": 5,
    "final_seeds": 2,
    "cv_folds": 2,
}
```

In `main`, replace the `final_top_k` option line with a `cv_folds` option (place it after `final_seeds`):

```python
    final_seeds: int | None = typer.Option(None, "--final-seeds"),
    cv_folds: int | None = typer.Option(None, "--cv-folds"),
```

Replace the smoke-unpacking and the dry-run echo and the `run_dataset` call body:

```python
    ds: list[str] = (
        _SMOKE["datasets"] if smoke else [d for d in datasets.split(",") if d]
    )
    nt: int | None = _SMOKE["n_trials"] if smoke else n_trials
    ep: int = _SMOKE["epochs"] if smoke else epochs
    fseeds: int | None = _SMOKE["final_seeds"] if smoke else final_seeds
    cvf: int | None = _SMOKE["cv_folds"] if smoke else cv_folds
    flavs = _parse_flavors(flavors)
    flav_names = [flavor_name(m, r) for m, r in flavs]

    if dry_run:
        typer.echo(
            f"would run datasets={ds} flavors={flav_names} backend={backend} "
            f"n_trials={nt} epochs={ep} n_jobs={n_jobs} cv_folds={cvf}"
        )
        raise typer.Exit(0)

    for dataset in ds:
        paths = run_dataset(
            dataset,
            backend=backend,
            flavors=flavs,
            n_trials=nt,
            epochs=ep,
            n_jobs=n_jobs,
            final_seeds=range(fseeds) if fseeds is not None else None,
            n_splits=cvf,
            out_dir=out_dir,
            storage_dir=storage_dir,
        )
        typer.echo(f"{dataset}: wrote {len(paths)} result files")
```

Confirm `final_top_k` appears nowhere: `grep -n final_top_k benchmarks/search.py` returns nothing.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/benchmarks/test_search_cli.py -q`
Expected: PASS (all four CLI tests).

- [ ] **Step 5: Lint + types**

Run: `uv run ruff check benchmarks/search.py && uv run --group bench mypy`
Expected: clean.

- [ ] **Step 6: Commit (unsigned)**

```bash
git add benchmarks/search.py tests/benchmarks/test_search_cli.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): CLI --cv-folds replaces --final-top-k"
```

---

### Task 6: Docs — protocol page + reframing

**Files:**
- Create: `docs/benchmarks/protocol.md`
- Modify: `docs/benchmarks/index.md` (toctree + Sections bullet)
- Modify: `docs/benchmarks/flavor-comparison.ipynb` (markdown wording + paper-row label)
- Modify: `docs/benchmarks/paper-reproduction/auto-mpg.ipynb` (markdown cell 0)
- Modify: `docs/benchmarks/paper-reproduction/tables.ipynb` (any "calibration anchor"/"wired correctly" wording)
- Modify: `benchmarks/README.md` (add a "Benchmark protocol" subsection)

**Interfaces:**
- Consumes: nothing in code. The flavor-comparison notebook already reads `test_metric`/`test_mean`/`test_std` (unchanged keys), so it still renders after Task 4; only labels/wording change.

- [ ] **Step 1: Create `docs/benchmarks/protocol.md`**

```markdown
# Benchmark protocol

`mononet`'s benchmarks use the standard held-out protocol for comparing tabular
models. For each dataset:

1. **Fixed splits.** We use the published `train_<ds>.csv` / `test_<ds>.csv`
   (Zenodo 10.5281/zenodo.7968969). The test set is touched exactly once, for the
   final report — never for any model-selection decision.
2. **Model selection on cross-validation only.** Hyperparameters and epochs are
   chosen on a *k*-fold cross-validation of the **train** split (stratified for
   classification). Folds: 5 for the small/medium datasets (Auto MPG, Heart,
   COMPAS); a single 80/20 holdout for the large ones (Loan, Blog), where a single
   split is already low-variance and *k*-fold would cost 5× for no real gain.
   The per-trial objective is the **mean metric across folds**.
3. **Refit + multi-seed test.** The single selected configuration is refit on the
   full train split and evaluated on the held-out test set across **10 seeds**
   (parameterisable).
4. **Reporting.** We report the **mean ± standard deviation over all seeds**. We do
   **not** select a best-*k* subset of seeds.

## Why our numbers differ from the original papers

The numbers quoted in Runje & Shankaranarayana (2023) and the prior baselines they
compared against were produced by a different protocol — inherited, via the
[`airtai/monotonic-nn`](https://github.com/airtai/monotonic-nn) reference code, from
those earlier papers. In that protocol the **test set is used as the validation
set**: hyperparameters are tuned with `validation_data=test`, early stopping
monitors the test loss, the per-run score is the **best epoch on the test curve**,
and the reported figure is the **mean of the best 5 of 10 runs**.

That makes those numbers optimistic by construction — the test set drives model
selection. Our protocol never lets the test set influence any choice, so our
held-out results sit somewhat **higher (worse)** than the published figures. The
difference is expected and is **not** a regression in `mononet`; the two sets of
numbers are simply **not directly comparable**. We keep the published figures in the
comparison tables for reference, labelled `[prior protocol]`.
```

- [ ] **Step 2: Wire into the toctree**

In `docs/benchmarks/index.md`, add a Sections bullet after the Overview bullet:

```markdown
- [Protocol](protocol.md) — how we train, select, and report; and why our numbers
  differ from the original papers.
```

and add `protocol` as the first entry of the hidden toctree:

```markdown
```{toctree}
:hidden:
:maxdepth: 2

protocol
00-overview
paper-reproduction/index
flavor-comparison
```
```

- [ ] **Step 3: Reframe `flavor-comparison.ipynb`**

Use NotebookEdit (or edit the ipynb JSON). In the **markdown cell (cell 0)**, change the line
`- One row per flavor (tuned best from Phase 2a search), with mean ± std across CV folds.`
to
`- One row per flavor (tuned best from Phase 2a search), with **test** mean ± std across seeds.`
and change `A `paper (CMNN) [quoted]` row with the number reported in Table 1 / Table 2 of` to
`A `paper (CMNN) [prior protocol]` row with the number reported in Table 1 / Table 2 of`,
then append a sentence:
`These paper numbers were obtained with a test-selected protocol and are **not directly comparable** to our held-out results — see [Protocol](protocol.md).`

In the **code cell (cell 1)**, change the single label string:
`rows.append({"method": "paper (CMNN) [quoted]", "value": pval, "std": "-"})`
to
`rows.append({"method": "paper (CMNN) [prior protocol]", "value": pval, "std": "-"})`

(Do not change the `PAPER` dict values or any `test_*` key reads.)

- [ ] **Step 4: Fix the Phase-1 "calibration anchor" wording**

In `docs/benchmarks/paper-reproduction/auto-mpg.ipynb`, markdown cell 0, replace:
`**Metric:** MSE (paper reports ≈ 8.37 for the CMNN baseline; this is the calibration anchor for checking that the harness is wired correctly).`
with:
`**Metric:** MSE. The paper reports ≈ 8.37 for the CMNN baseline, but under a test-selected protocol (HP search, early stopping, and best-epoch all on the test set, then best-5-of-10). A correctly-wired *held-out* harness should **not** reproduce 8.37 — it reports a somewhat higher, honest number. See [Protocol](../protocol.md).`

In `docs/benchmarks/paper-reproduction/tables.ipynb`, search for any "calibration anchor" or "wired correctly" phrasing (`grep -l` the JSON) and apply the same reframing; if none is present, leave it unchanged.

- [ ] **Step 5: README subsection**

In `benchmarks/README.md`, add a `## Benchmark protocol` subsection (place it above the Phase-2a runbook section) summarising the 4 steps and linking to `../docs/benchmarks/protocol.md`:

```markdown
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
```

- [ ] **Step 6: Verify docs build + spelling**

Run: `uv run pre-commit run --all-files --hook-stage manual` (codespell, end-of-file-fixer) then `./tools/build-docs.sh`
Expected: pre-commit clean; Sphinx build succeeds with `protocol` in the benchmarks toctree and no warnings-as-errors failure.

- [ ] **Step 7: Commit (unsigned)**

```bash
git add docs/benchmarks/protocol.md docs/benchmarks/index.md docs/benchmarks/flavor-comparison.ipynb docs/benchmarks/paper-reproduction/auto-mpg.ipynb docs/benchmarks/paper-reproduction/tables.ipynb benchmarks/README.md
git -c commit.gpgsign=false commit -m "docs(benchmarks): protocol page + reframe paper numbers as prior-protocol"
```

---

### Task 7: Re-run AutoMPG under the new protocol (controller-executed)

> **Not a TDD unit** — this is a long CPU job plus PR/signing operations the controller runs after Tasks 1–6 pass review and the whole-branch review is clean. Do not dispatch a fresh subagent for it.

**Files:**
- Regenerate: `benchmarks/results/phase2/auto-{switch,absolute}-{plain,residual}.json`
- Re-render: `docs/benchmarks/flavor-comparison.ipynb`

- [ ] **Step 1: Re-run the full AutoMPG study under the new protocol**

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 tools/mononet-benchmark-search \
  --datasets auto --backend torch --n-jobs 6
```
Uses `_BUDGET["auto"]` = 50 trials, 10 seeds, **5-fold CV**. Writes the four `auto-*.json`.

- [ ] **Step 2: Verify finiteness + schema**

```bash
uv run python - <<'PY'
import json, math
from pathlib import Path
for f in sorted(Path("benchmarks/results/phase2").glob("auto-*.json")):
    r = json.loads(f.read_text())
    assert {"cv_best","test_mean","test_std","n_seeds"} <= set(r)
    assert "n_selected" not in r and "val_best" not in r
    assert all(math.isfinite(r[k]) for k in ("cv_best","test_mean","test_std"))
    print(f"{r['flavor']:18s} cv_best={r['cv_best']:.3f}  test={r['test_mean']:.3f}±{r['test_std']:.3f}  seeds={r['n_seeds']}")
PY
```
Expected: 4 lines, all finite, `n_seeds == 10`, MSE in the single-to-low-double-digit band.

- [ ] **Step 3: Re-render the comparison notebook**

```bash
uv run --group bench --group docs --extra torch jupyter nbconvert \
  --to notebook --execute --inplace docs/benchmarks/flavor-comparison.ipynb
```
Expected: executes cleanly; the auto row shows the 4 flavors + `paper (CMNN) [prior protocol]` + XGBoost.

- [ ] **Step 4: Commit the regenerated artifacts (unsigned)**

```bash
git add benchmarks/results/phase2/auto-*.json docs/benchmarks/flavor-comparison.ipynb
git -c commit.gpgsign=false commit -m "bench(auto): regenerate AutoMPG numbers under standard protocol"
```

- [ ] **Step 5: Re-sign the whole branch, push, retitle PR**

```bash
# Re-sign every commit added on this branch (controller; approve Touch ID prompts):
git rebase --exec "git commit --amend --no-edit -n -S" $(git merge-base main HEAD)
git log --format="%h %G? %s" $(git merge-base main HEAD)..HEAD   # expect all G
git push --force-with-lease
gh pr edit 62 --title "Adopt standard benchmark protocol + honest AutoMPG numbers (+ search-wrapper fix)"
```

- [ ] **Step 6: Confirm CI green**

```bash
gh pr checks 62
```
Expected: all checks pass (15 test legs + static-analysis + pre-commit + docs-smoke).

---

## Notes for the executor

- Run mypy as `uv run --group bench mypy` (the canonical CI command); a bare `uv run mypy` lacks typer/optuna and will false-fail on `benchmarks/`.
- The synthetic test bundles are tiny; keep search tests at `n_splits=2` and `epochs=1` so CI stays fast.
- After Tasks 1–6, the whole-branch review runs; only then does the controller execute Task 7.
