# Deep Monotonic Residual — Real-Dataset Accuracy (Stage 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `{mode}-deep` flavor (residual stack with a larger depth search band, `sub_depth=2` fixed) to the benchmark flavor study — **plumbing only**; the heavy GPU search runs in a follow-up session per a committed runbook.

**Architecture:** A deep flavor is *just* `residual=True` with a larger depth band, so `model_builder.py` is untouched. Flavors become triples `(mode, residual, deep)`; the depth band branch lives in `suggest_config`; naming/bookkeeping lives in `flavor_name`/`_ALL_FLAVORS`/CLI. A docs skeleton + runbook hand the real run to the GPU session.

**Tech Stack:** Python 3.11+, Optuna (TPE), Typer CLI, pytest, torch/jax/keras backends (via `benchmarks/_common`), Sphinx + myst-nb docs.

## Global Constraints

- Work on branch `feat/deep-residual-accuracy` (already created off latest `main`, through #69). **Never commit to `main`.**
- Commit **UNSIGNED** during subagent execution: `git -c commit.gpgsign=false commit`. Controller re-signs the whole branch before push.
- Per-task gates (all must pass): `uv run ruff check`, `uv run ruff format --check`, `uv run --group bench mypy`, the task's pytest.
- Final gates: `uv run pre-commit run --all-files --hook-stage manual` + `./tools/build-docs.sh`.
- No Pydantic; stdlib dataclasses only. MyST field-list docstrings on public functions/classes. ruff line-length 88. Strict mypy. Preserve lazy backend imports. `benchmarks/` is never shipped in the wheel. Result JSON written with a trailing newline. Never commit `*.db`/`*.jsonl`.
- **`benchmarks/_common/model_builder.py` MUST NOT change.** `sub_depth` stays the `MonoResidual` default (2) and is **not searched**.
- **NO real search in this session / in CI** — only tiny synthetic smoke tests. The docs results page is a **skeleton** with a placeholder table (plain Markdown, no `{code-cell}`); the GPU session fills it.
- Backend tests use `pytest.importorskip`; benchmark tests import `optuna`/`torch` via `importorskip` (add `# noqa: E402` after import-skip lines, matching existing files).

---

### Task 1: `suggest_config` deep depth band

**Files:**
- Modify: `benchmarks/_common/search_spaces.py` (the `suggest_config` function, currently lines 13–66)
- Test: `tests/benchmarks/test_search_spaces.py` (extend; currently 63 lines)

**Interfaces:**
- Consumes: `optuna.Trial`, `BenchmarkConfig`, `OptimizerSpec` (unchanged).
- Produces: `suggest_config(trial, *, dataset, backend, mode, residual, epochs, metric, deep: bool = False) -> BenchmarkConfig`. When `deep=True`, `depth ∈ {6,10,16}`; else `depth ∈ [1,4]`. All other sampled fields unchanged.

- [ ] **Step 1: Write the failing tests**

Add to `tests/benchmarks/test_search_spaces.py` (the `_cfg` helper already exists; add a `deep` passthrough to it and two new tests):

```python
def _cfg(
    mode: Literal["switch", "absolute"],
    residual: bool,
    metric: Literal["accuracy", "rmse", "mse"] = "mse",
    deep: bool = False,
) -> BenchmarkConfig:
    study = optuna.create_study()
    trial = study.ask()
    return suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode=mode,
        residual=residual,
        epochs=3,
        metric=metric,
        deep=deep,
    )


def test_deep_samples_depth_from_high_band() -> None:
    # Deep flavor draws depth from the categorical high band, never the 1..4 range.
    for _ in range(25):
        cfg = _cfg("absolute", residual=True, deep=True)
        assert cfg.depth in (6, 10, 16)
        assert cfg.residual is True
        assert 0.0 <= cfg.convex_fraction <= 1.0  # other fields still sampled


def test_non_deep_keeps_shallow_depth_band() -> None:
    for _ in range(25):
        cfg = _cfg("switch", residual=True, deep=False)
        assert 1 <= cfg.depth <= 4
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/benchmarks/test_search_spaces.py::test_deep_samples_depth_from_high_band -v`
Expected: FAIL — `suggest_config() got an unexpected keyword argument 'deep'`.

- [ ] **Step 3: Implement the deep band**

In `benchmarks/_common/search_spaces.py`, add the `deep` parameter and branch. The signature becomes (note the new final keyword-only param and its docstring entry):

```python
def suggest_config(
    trial: optuna.Trial,
    *,
    dataset: str,
    backend: Literal["torch", "jax", "keras"],
    mode: Literal["switch", "absolute"],
    residual: bool,
    epochs: int,
    metric: Literal["accuracy", "rmse", "mse"],
    deep: bool = False,
) -> BenchmarkConfig:
```

Add to the docstring's field list (after the `metric` entry, before `:returns:`):

```
    :param deep: When ``True``, draw ``depth`` from the deep categorical band
        ``{6, 10, 16}`` (residual skips make these trainable); otherwise draw
        ``depth`` from the shallow range ``[1, 4]``. Only affects the ``depth``
        dimension; all other hyperparameters are sampled identically.
```

Replace the single `depth` line:

```python
    width = trial.suggest_categorical("width", [8, 16, 21, 32, 64])
    if deep:
        depth = trial.suggest_categorical("depth", [6, 10, 16])
    else:
        depth = trial.suggest_int("depth", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
```

Leave everything else (width, lr, weight_decay, dropout, lr_decay, batch_size, convex_fraction, and the returned `BenchmarkConfig(...)`) exactly as-is.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_search_spaces.py -v`
Expected: PASS (the two new tests plus the two pre-existing ones).

- [ ] **Step 5: Gates + commit**

```bash
uv run ruff check && uv run ruff format --check && uv run --group bench mypy
git add benchmarks/_common/search_spaces.py tests/benchmarks/test_search_spaces.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): deep depth band in suggest_config (Stage 2)"
```

---

### Task 2: Flavor triples + thread `deep` through the search engine

**Files:**
- Modify: `benchmarks/_common/search.py` (`flavor_name` L26–27, `search` L70–121, `_ALL_FLAVORS` L165–170, `run_dataset` L183–253)
- Test: `tests/benchmarks/test_search.py` (extend), `tests/benchmarks/test_run_dataset.py` (update the 2-tuple flavors to 3-tuples; add a deep smoke)

**Interfaces:**
- Consumes: `suggest_config(..., deep=deep)` from Task 1; `flavor_name(mode, residual, deep=False)`.
- Produces:
  - `flavor_name(mode: str, residual: bool, deep: bool = False) -> str` — `"{mode}-deep"` when `deep`, else `"{mode}-{'residual' if residual else 'plain'}"`.
  - `_ALL_FLAVORS: tuple[tuple[str, bool, bool], ...]` — 6 triples.
  - `search(..., deep: bool = False)` — objective uses `deep`; `study_name`/`StudyResult.flavor` = `flavor_name(mode, residual, deep)`.
  - `run_dataset(..., flavors: tuple[tuple[str, bool, bool], ...] = _ALL_FLAVORS)` — loop `for mode, residual, deep in flavors:`, threads `deep` into `search`; `final_eval` is called unchanged (no `deep`).

- [ ] **Step 1: Write the failing tests**

Update `tests/benchmarks/test_search.py::test_flavor_name` and add a deep case:

```python
def test_flavor_name() -> None:
    assert flavor_name("switch", False) == "switch-plain"
    assert flavor_name("absolute", True) == "absolute-residual"
    assert flavor_name("switch", True, deep=True) == "switch-deep"
    assert flavor_name("absolute", True, deep=True) == "absolute-deep"


def test_all_flavors_has_six_entries_including_deep() -> None:
    from benchmarks._common.search import _ALL_FLAVORS, flavor_name

    assert len(_ALL_FLAVORS) == 6
    names = {flavor_name(m, r, d) for m, r, d in _ALL_FLAVORS}
    assert {"switch-deep", "absolute-deep"} <= names


def test_search_deep_flavor_names_study_and_uses_high_depth() -> None:
    res = search(
        _bundle(),
        mode="absolute",
        residual=True,
        deep=True,
        backend="torch",
        n_trials=2,
        seed=0,
        epochs=1,
        n_splits=2,
    )
    assert res.flavor == "absolute-deep"
    assert res.best_params["depth"] in (6, 10, 16)
```

In `tests/benchmarks/test_run_dataset.py`, update the existing flavors argument (2-tuples → 3-tuples) and add a deep smoke test:

```python
def test_run_dataset_writes_one_json_per_flavor(tmp_path: Path) -> None:
    paths = run_dataset(
        "auto",
        backend="torch",
        flavors=(("switch", False, False), ("absolute", False, False)),
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
    # ... (rest of the existing assertions unchanged)


def test_run_dataset_deep_flavor_writes_deep_json(tmp_path: Path) -> None:
    import json as _json
    import math

    paths = run_dataset(
        "auto",
        backend="torch",
        flavors=(("absolute", True, True),),
        n_trials=2,
        epochs=1,
        final_seeds=range(2),
        n_splits=2,
        data_dir=FIXTURES,
        out_dir=tmp_path,
    )
    assert [p.name for p in paths] == ["auto-absolute-deep.json"]
    rec = _json.loads(paths[0].read_text())
    assert rec["flavor"] == "absolute-deep"
    assert rec["best_params"]["depth"] in (6, 10, 16)
    assert math.isfinite(rec["test_mean"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/benchmarks/test_search.py::test_flavor_name tests/benchmarks/test_run_dataset.py::test_run_dataset_deep_flavor_writes_deep_json -v`
Expected: FAIL — `flavor_name()` takes 2 positional args / `search()` got unexpected kwarg `deep` / `run_dataset` cannot unpack 3-tuple.

- [ ] **Step 3: Implement `flavor_name` + `_ALL_FLAVORS`**

Replace `flavor_name` (L26–27):

```python
def flavor_name(mode: str, residual: bool, deep: bool = False) -> str:
    """Canonical flavor label for result files and Optuna study names.

    :param mode: Monotonicity mode (``"switch"`` or ``"absolute"``).
    :param residual: Whether the stack uses residual blocks.
    :param deep: Whether this is the deep-depth-band flavor. When ``True`` the
        label is ``"{mode}-deep"`` regardless of ``residual`` (deep implies
        residual); otherwise ``"{mode}-residual"`` or ``"{mode}-plain"``.
    :returns: The flavor label string.
    """
    if deep:
        return f"{mode}-deep"
    return f"{mode}-{'residual' if residual else 'plain'}"
```

Replace `_ALL_FLAVORS` (L165–170):

```python
# (mode, residual, deep) triples. Deep implies residual=True with a larger
# depth search band (see suggest_config); it is a separate Optuna study.
_ALL_FLAVORS: tuple[tuple[str, bool, bool], ...] = (
    ("switch", False, False),
    ("switch", True, False),
    ("absolute", False, False),
    ("absolute", True, False),
    ("switch", True, True),
    ("absolute", True, True),
)
```

- [ ] **Step 4: Thread `deep` through `search`**

In `search` (L70–121): add `deep: bool = False` to the keyword-only params (after `residual: bool`). This function currently has only a one-line summary docstring and no `:param:` field list, so match that existing style — do not add a field list for `deep`. Pass `deep=deep` into `suggest_config`, and use `flavor_name(mode, residual, deep)` in BOTH the `study_name` and the returned `StudyResult.flavor`:

```python
def search(
    bundle: DatasetBundle,
    *,
    mode: str,
    residual: bool,
    backend: str,
    deep: bool = False,
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
            deep=deep,
        )
        scores: list[float] = []
        for fb in folds:
            rows = run(cfg, fb)
            if not rows:
                raise RuntimeError("run() returned no rows for trial")
            scores.append(float(rows[0].scores[metric]))  # type: ignore[index]
        return float(np.mean(scores))

    study = optuna.create_study(
        study_name=f"{bundle.name}-{flavor_name(mode, residual, deep)}",
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return StudyResult(
        dataset=bundle.name,
        flavor=flavor_name(mode, residual, deep),
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
    )
```

Leave `final_eval` (L124–162) **unchanged** — it builds its config from `best_params["depth"]` + `residual`, so it needs no `deep` param.

- [ ] **Step 5: Thread `deep` through `run_dataset`**

In `run_dataset` (L183–253): change the `flavors` annotation/default to triples and the loop to unpack three values; pass `deep` to `search`; keep `final_eval` call unchanged; derive the `.db` filename and JSON name from `study.flavor`:

```python
def run_dataset(
    dataset: str,
    *,
    backend: str = "torch",
    flavors: tuple[tuple[str, bool, bool], ...] = _ALL_FLAVORS,
    n_trials: int | None = None,
    epochs: int = 50,
    n_jobs: int = 1,
    final_seeds: Iterable[int] | None = None,
    n_splits: int | None = None,
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    storage_dir: Path | None = None,
) -> list[Path]:
```

Loop body (replaces L215–252):

```python
    for mode, residual, deep in flavors:
        fname = flavor_name(mode, residual, deep)
        storage = (
            f"sqlite:///{storage_dir}/{dataset}-{fname}.db" if storage_dir else None
        )
        study = search(
            bundle,
            mode=mode,
            residual=residual,
            deep=deep,
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

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_search.py tests/benchmarks/test_run_dataset.py -v`
Expected: PASS (including the pre-existing `test_storage_uses_deterministic_study_name_so_it_resumes`, which calls `flavor_name('switch', False)` — still valid with `deep` defaulting `False`).

- [ ] **Step 7: Gates + commit**

```bash
uv run ruff check && uv run ruff format --check && uv run --group bench mypy
git add benchmarks/_common/search.py tests/benchmarks/test_search.py tests/benchmarks/test_run_dataset.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): {mode}-deep flavor triples threaded through search engine"
```

---

### Task 3: CLI — parse & default the deep flavor

**Files:**
- Modify: `benchmarks/search.py` (`_parse_flavors` L28–46, the `main` command body where `flav_names` is built ~L75–76)
- Test: `tests/benchmarks/test_search_cli.py` (extend)

**Interfaces:**
- Consumes: `_ALL_FLAVORS` (6 triples), `flavor_name(mode, residual, deep)` from Task 2.
- Produces: `_parse_flavors(spec) -> tuple[tuple[str, bool, bool], ...]`, accepting `kind ∈ {plain, residual, deep}` (`deep → (mode, True, True)`, `residual → (mode, True, False)`, `plain → (mode, False, False)`); the `main` default flavor set is all 6; `flav_names` built by unpacking triples.

- [ ] **Step 1: Write the failing tests**

Add to `tests/benchmarks/test_search_cli.py`:

```python
def test_parse_flavors_accepts_deep() -> None:
    from benchmarks.search import _parse_flavors

    assert _parse_flavors("absolute-deep") == (("absolute", True, True),)
    assert _parse_flavors("switch-plain,switch-residual") == (
        ("switch", False, False),
        ("switch", True, False),
    )


def test_parse_flavors_default_is_all_six() -> None:
    from benchmarks.search import _parse_flavors

    assert len(_parse_flavors("")) == 6


def test_dry_run_lists_deep_flavor() -> None:
    res = runner.invoke(app, ["--flavors", "absolute-deep", "--dry-run"])
    assert res.exit_code == 0
    assert "absolute-deep" in res.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/benchmarks/test_search_cli.py::test_parse_flavors_accepts_deep -v`
Expected: FAIL — `_parse_flavors` returns 2-tuples / `deep` rejected as a bad flavor.

- [ ] **Step 3: Implement `_parse_flavors` + `flav_names`**

Replace `_parse_flavors` (L28–46):

```python
def _parse_flavors(spec: str | None) -> tuple[tuple[str, bool, bool], ...]:
    """Parse a comma-separated flavor spec into ``(mode, residual, deep)`` triples.

    :param spec: Comma-separated flavor names like ``switch-plain,absolute-deep``,
        or ``None``/empty string to return all flavors.
    :returns: Tuple of ``(mode, residual, deep)`` triples.
    """
    if not spec:
        return _ALL_FLAVORS
    valid_modes = {"switch", "absolute"}
    valid_kinds = {"plain", "residual", "deep"}
    out: list[tuple[str, bool, bool]] = []
    for name in spec.split(","):
        mode, _, kind = name.partition("-")
        if mode not in valid_modes or kind not in valid_kinds:
            raise typer.BadParameter(f"bad flavor: {name}")
        out.append((mode, kind in ("residual", "deep"), kind == "deep"))
    return tuple(out)
```

Add the import of `_ALL_FLAVORS` to the existing import line (L14):

```python
from benchmarks._common.search import _ALL_FLAVORS, flavor_name, run_dataset
```

Update the `flav_names` construction (currently `flav_names = [flavor_name(m, r) for m, r in flavs]`, ~L76):

```python
    flav_names = [flavor_name(m, r, d) for m, r, d in flavs]
```

The `run_dataset(..., flavors=flavs, ...)` call already forwards `flavs`; no other change needed (it now carries triples).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_search_cli.py -v`
Expected: PASS (new tests + the four pre-existing CLI tests, including `test_invalid_flavors_exits_nonzero`).

- [ ] **Step 5: Gates + commit**

```bash
uv run ruff check && uv run ruff format --check && uv run --group bench mypy
git add benchmarks/search.py tests/benchmarks/test_search_cli.py
git -c commit.gpgsign=false commit -m "feat(benchmarks): CLI parses and defaults the deep flavor"
```

---

### Task 4: Docs skeleton page + wiring

**Files:**
- Create: `docs/benchmarks/deep-residual-accuracy.md`
- Modify: `docs/benchmarks/index.md` (Sections list + `{toctree}`), `docs/concepts/monotonic-residual.md` (cross-link from the "Real-dataset accuracy (forthcoming)" section)

**Interfaces:**
- Consumes: nothing at build time (plain Markdown; the results table is a placeholder skeleton filled later by the GPU session from `results/phase2/*-deep.json`).
- Produces: a benchmarks doc page wired into the toctree; a cross-link from the concepts residual page.

- [ ] **Step 1: Create `docs/benchmarks/deep-residual-accuracy.md`**

```markdown
# Deep monotonic residual — real-dataset accuracy

**Status: results pending the GPU search run.** This page reports whether the
now-trainable *deep* monotonic residual stacks (`MonoResidual` with
`sub_depth=2` skips — see [the residual construction](../concepts/monotonic-residual.md))
improve held-out test accuracy over the shallow tuned flavors, across the five
benchmark datasets.

## Question

Stage 1 showed residual skips make depth-32 monotone stacks *trainable* on
synthetic data. This study measures whether that trainability translates into
better test metrics on real tabular data, under the
[standard benchmark protocol](protocol.md) (5-fold CV model selection for the
small/medium datasets, single holdout for the large ones; mean ± std over all
final seeds; the test set is touched once).

A **null or negative result** — depth not improving, or mildly hurting, accuracy
on these small/medium tabular datasets — is an expected and reported outcome.
Stage 1 establishes the capability; Stage 2 measures whether it pays off.

## Flavors

Six flavors per dataset: `{switch, absolute} × {plain, residual, deep}`. The
**deep** flavor is a residual stack (`sub_depth=2`) whose depth is searched over
`{6, 10, 16}` blocks (effective ≈ 14 / 22 / 34 layers); plain/residual search
`depth ∈ [1, 4]`. All other hyperparameters share one search space, so depth is
the only structural difference between `residual` and `deep`.

## Results

_Results pending the GPU search run (see the reproduce command below). Test
metric is MSE for `auto`, RMSE for `blog`, accuracy for `heart`/`compas`/`loan`
(lower is better for MSE/RMSE; higher for accuracy)._

| dataset | metric | best shallow (mode) | deep (mode) | Δ | deep depth |
|---|---|---|---|---|---|
| auto | MSE | — | — | — | — |
| heart | accuracy | — | — | — | — |
| compas | accuracy | — | — | — | — |
| loan | accuracy | — | — | — | — |
| blog | RMSE | — | — | — | — |

## Reproduce

```
uv run --extra torch --group bench python -m benchmarks.search \
    --datasets auto,heart,compas,loan,blog
```

This runs all six flavors per dataset and writes
`benchmarks/results/phase2/<dataset>-<flavor>.json`. See
[`benchmarks/RUNBOOK-stage2.md`](https://github.com/davorrunje/mononet/blob/main/benchmarks/RUNBOOK-stage2.md)
for the full GPU run procedure.
```

- [ ] **Step 2: Wire into `docs/benchmarks/index.md`**

In the `## Sections` bullet list, add after the Flavor-comparison bullet:

```markdown
- [Deep residual accuracy](deep-residual-accuracy.md) — does the now-trainable
  depth (residual `sub_depth=2` skips) improve real-dataset test accuracy over
  the shallow tuned flavors?
```

In the `{toctree}` block, add `deep-residual-accuracy` after `flavor-comparison`:

```
protocol
00-overview
paper-reproduction/index
flavor-comparison
deep-residual-accuracy
deep-init
```

- [ ] **Step 3: Cross-link from `docs/concepts/monotonic-residual.md`**

The page has a `## Real-dataset accuracy (forthcoming)` section. Replace its body with a cross-link (keep the heading text or rename to drop "forthcoming"):

```markdown
## Real-dataset accuracy

Whether this now-trainable depth improves held-out accuracy on real datasets —
versus the shallow tuned flavors — is measured in
[Deep residual accuracy](../benchmarks/deep-residual-accuracy.md) (Stage 2).
```

- [ ] **Step 4: Verify the docs build**

Run: `./tools/build-docs.sh`
Expected: `build succeeded`; `deep-residual-accuracy` appears in the benchmarks toctree output; no new warnings about the cross-links (both targets exist).

- [ ] **Step 5: Gates + commit**

```bash
uv run pre-commit run --all-files --hook-stage manual
git add docs/benchmarks/deep-residual-accuracy.md docs/benchmarks/index.md docs/concepts/monotonic-residual.md
git -c commit.gpgsign=false commit -m "docs(benchmarks): deep-residual-accuracy skeleton + toctree + cross-link"
```

---

### Task 5: `RUNBOOK-stage2.md` handoff

**Files:**
- Create: `benchmarks/RUNBOOK-stage2.md`

**Interfaces:**
- Consumes: the deep-flavor CLI (Task 3), the docs skeleton (Task 4).
- Produces: an end-to-end procedure the GPU session follows to run the search, commit results, fill the docs table, and open the follow-up PR.

- [ ] **Step 1: Create `benchmarks/RUNBOOK-stage2.md`**

```markdown
# Stage 2 GPU run — deep monotonic residual accuracy

This runbook executes the Stage-2 accuracy study on a GPU machine
(`gpu-torch` devcontainer, 5090 / Blackwell sm_120). The plumbing (deep flavor,
CLI, docs skeleton) is already merged; this run produces the numbers.

## 0. Environment

Open the repo in the **`gpu-torch`** devcontainer flavor (see
`.devcontainer/`). Then:

```bash
uv sync --extra torch --group bench
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and the 5090 device name.

## 1. Run the full search

Six flavors per dataset (`{switch,absolute} × {plain,residual,deep}`), default
per-dataset budgets (`auto/heart/compas`: 50 trials, 10 seeds, 5-fold CV;
`loan/blog`: 25 trials, 5 seeds, single holdout):

```bash
uv run --extra torch --group bench python -m benchmarks.search \
    --datasets auto,heart,compas,loan,blog \
    --storage-dir benchmarks/results/deep-residual-accuracy/studies
```

Notes:
- Only `auto` had results under the standard protocol; heart/compas/loan/blog
  shallow flavors are (re)generated here too.
- The `sub_depth=2` default (merged in #67) makes the *prior* `auto` residual
  numbers stale — this run regenerates all `auto` flavors, so the whole table
  is internally consistent.
- `--storage-dir` writes resumable Optuna `.db` files; these are git-ignored
  (never commit `*.db`). Use `--n-jobs` to parallelize trials on the GPU box.
- To smoke-test the plumbing first: append `--smoke` (tiny 2-trial/2-fold run).

Outputs: `benchmarks/results/phase2/<dataset>-<flavor>.json` for all 5 × 6 = 30
files (four `auto-*` are overwritten; the rest are new).

## 2. Commit the results

```bash
git checkout -b feat/deep-residual-accuracy-results
git add benchmarks/results/phase2/*.json
git commit -S -m "bench(results): Stage 2 deep-vs-shallow accuracy (all 5 datasets, 6 flavors)"
```

(Confirm no `*.db`/`*.jsonl` are staged.)

## 3. Fill the docs table

Edit `docs/benchmarks/deep-residual-accuracy.md` — replace the placeholder
`—` cells in the Results table. For each dataset:
- **metric**: MSE (`auto`), RMSE (`blog`), accuracy (`heart`/`compas`/`loan`).
- **best shallow (mode)**: the better of the four shallow flavors
  (`{switch,absolute}-{plain,residual}`) by `test_mean` (min for MSE/RMSE, max
  for accuracy); note which mode won.
- **deep (mode)**: the better of `{switch,absolute}-deep` by `test_mean`.
- Report each as `test_mean ± test_std`; **Δ** = deep − best-shallow (sign per
  the metric's direction — note whether deep helped).
- **deep depth**: `best_params["depth"]` of the reported deep flavor.

Remove the "Status: results pending" banner and the "Results pending" italic
note once filled. Then:

```bash
./tools/build-docs.sh   # expect: build succeeded, no new warnings
```

## 4. Open the follow-up PR

```bash
git push -u origin feat/deep-residual-accuracy-results
gh pr create --base main \
    --title "bench: Stage 2 deep-vs-shallow monotonic residual accuracy" \
    --body "Fills the deep-residual-accuracy results table from the GPU search run. Closes the Stage 2 follow-up."
```

All commits must be signed. If tool-driven signing is flaky, commit unsigned
and re-sign before push:
`git rebase --exec "git commit --amend --no-edit -n -S" $(git merge-base main HEAD)`.
```

- [ ] **Step 2: Verify + commit**

```bash
uv run pre-commit run --all-files --hook-stage manual
git add benchmarks/RUNBOOK-stage2.md
git -c commit.gpgsign=false commit -m "docs(benchmarks): Stage 2 GPU run runbook"
```

(`pre-commit` here mainly checks trailing whitespace / EOF / codespell on the new Markdown; reword rather than editing the codespell config if it trips.)

---

## Controller-executed final phase (NOT a subagent task)

After all tasks pass their reviews:

1. Re-sign the whole branch: `git rebase --exec "git commit --amend --no-edit -n -S" $(git merge-base main HEAD)` (user present for Touch ID; if signing is refused mid-run, retry).
2. Verify: `git log --format="%G? %h %s" $(git merge-base main HEAD)..HEAD` shows `G` for every commit.
3. Push: `git push -u origin feat/deep-residual-accuracy`.
4. Open a **DRAFT** PR (`gh pr create --draft`) whose body: (a) summarizes the plumbing (deep flavor, no model_builder change, sub_depth=2 fixed); (b) states the results are produced by a **follow-up GPU session** per `benchmarks/RUNBOOK-stage2.md`; (c) explicitly instructs that session to run the search, commit results, fill the docs table, and undraft/merge.
5. **Do NOT run the real search in this session.**

---

## Notes

- `final_eval` deliberately takes no `deep` param: its config is fully
  determined by `residual` + `best_params["depth"]`; adding `deep` would be a
  dead argument (see spec §3.2).
- `model_builder.py` is intentionally untouched — a deep config is a residual
  config with a larger `depth`, and the residual branch already defaults
  `MonoResidual` to `sub_depth=2`.
- The four committed `auto-*` result JSONs are NOT regenerated in this session
  (no real search here); the GPU run regenerates them for table consistency.
