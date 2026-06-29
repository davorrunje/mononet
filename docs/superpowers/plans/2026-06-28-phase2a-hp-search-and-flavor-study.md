# Phase 2a — HP-Search Engine + Flavor Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an Optuna-based hyperparameter-search engine that drives the existing `run(cfg, bundle)` to tune each `{switch,absolute}×{plain,residual}` flavor per dataset, with a validation split that never touches test, plus search notebooks and a rendered flavor-comparison summary.

**Architecture:** Pure-Python engine in `benchmarks/_common/` (`splits.py`, `search_spaces.py`, `search.py`). Search reuses Phase-1's `run()` by swapping the held-out validation set into a throwaway bundle's `X_test` slot (`dataclasses.replace`) — no runner change. Final eval reuses `run()` on the real bundle + `aggregate()`. Real searches run from `benchmarks/notebooks/` (manual, repo-only); a `docs/` summary notebook renders committed result JSON.

**Tech Stack:** Optuna (TPE sampler), the Phase-1 harness (`run`, `BenchmarkConfig`, `aggregate`, `registry.load`, `DatasetBundle`), numpy, scikit-learn (split), pytest.

## Global Constraints

- Build on Phase 1 (merged): `run(cfg: BenchmarkConfig, bundle: DatasetBundle) -> list[ResultRow]` trains on `bundle.X_train` and evaluates on `bundle.X_test`; `aggregate(rows, *, metric, lower_is_better, top_k=5) -> Aggregate(metric, mean, std, n_seeds, n_selected)`; `registry.load(name, *, data_dir) -> DatasetBundle`; `BenchmarkConfig` fields per Phase 1; `OptimizerSpec(name, lr, weight_decay)`.
- `DatasetBundle` stays **unchanged** (frozen, slots). Validation is a **helper**, never a bundle field. Test data is never read during search.
- Four flavors: `mode ∈ {"switch","absolute"}` × `residual ∈ {False, True}`. Flavor string = `f"{mode}-{'residual' if residual else 'plain'}"`.
- `convex_fraction` is searched **only when `mode == "absolute"`**. `activation` is fixed `"elu"` (not searched in 2a).
- Search optimizes the **validation** metric; final numbers are **test**, reported as mean±std of **best-5-of-10** seeds after refitting best HPs on the full train split.
- Direction: minimize for `mse`/`rmse`, maximize for `accuracy`. Dataset primary metric: `accuracy` for binary, `mse` for regression (override allowed).
- `optuna` added to the `bench` group as a `==` pin (repo policy). Tests `pytest.importorskip("optuna")` and the active backend.
- Real 20-study searches + the committed `benchmarks/results/phase2/<dataset>-<flavor>.json` are **manual maintainer runs** — NOT CI. CI runs only fast synthetic tests.
- No `mononet` package change; `benchmarks/` stays out of the wheel.
- Branch: `feat/phase2a-hp-search` (already created, holds the spec). Never commit to `main`. Commits signed; end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer. Run commands from repo root `/Users/davor/Projects/PhD/mononet`.

> **Spec deviation to confirm at execution:** the spec §3 names a *median pruner on the
> per-epoch validation curve*. `run()` is single-shot (trains all epochs, evaluates once)
> and exposes no per-epoch hook, so per-epoch pruning would require threading a callback
> through all three backend training loops — disproportionate for 2a. This plan uses
> Optuna's **TPE sampler without a pruner** (full trials) and documents per-epoch pruning
> as a future enhancement. TPE sampling — the main efficiency win — is unaffected.

---

### Task 1: `optuna` dependency + `train_val_split` helper

**Files:**
- Modify: `pyproject.toml` (`bench` group += optuna)
- Create: `benchmarks/_common/splits.py`
- Test: `tests/benchmarks/test_splits.py`

**Interfaces:**
- Produces: `train_val_split(bundle, *, frac=0.2, seed, stratify=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` returning `(X_tr, y_tr, X_val, y_val)`.

- [ ] **Step 1: Add optuna to the bench group + lock**

In `pyproject.toml` `[dependency-groups].bench`, add an optuna `==` pin (after `xgboost`):

```toml
    "optuna==4.6.0",
```

Run `uv lock`; if `4.6.0` isn't what resolves on Python 3.11–3.13, use the resolved version and record it. Then `uv sync --group bench`. (Confirm `uv lock --check` passes.)

- [ ] **Step 2: Write the failing test** — `tests/benchmarks/test_splits.py`:

```python
import numpy as np

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.splits import train_val_split


def _bundle(task: str, n: int = 200, d: int = 5) -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d))
    if task == "binary_classification":
        y = (X[:, 0] > 0).astype(np.float64)
    else:
        y = X[:, 0] + 0.1 * rng.normal(size=n)
    Xt = rng.normal(size=(10, d))
    yt = np.zeros(10)
    return DatasetBundle(
        name="syn", task=task,  # type: ignore[arg-type]
        X_train=X, y_train=y, X_test=Xt, y_test=yt,
        mono_increasing=(0,), mono_decreasing=(),
        feature_names=tuple(f"f{i}" for i in range(d)), metadata={},
    )


def test_split_sizes_sum_to_train() -> None:
    b = _bundle("regression")
    X_tr, y_tr, X_val, y_val = train_val_split(b, frac=0.2, seed=0)
    assert len(X_tr) + len(X_val) == len(b.X_train)
    assert len(X_val) == 40 and len(X_tr) == 160
    assert len(y_tr) == 160 and len(y_val) == 40


def test_stratified_preserves_class_balance() -> None:
    b = _bundle("binary_classification")
    _, y_tr, _, y_val = train_val_split(b, frac=0.25, seed=0)
    # both splits contain both classes
    assert set(np.unique(y_tr)) == {0.0, 1.0}
    assert set(np.unique(y_val)) == {0.0, 1.0}


def test_deterministic_for_seed() -> None:
    b = _bundle("regression")
    a = train_val_split(b, seed=3)
    c = train_val_split(b, seed=3)
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[2], c[2])


def test_does_not_return_test_data() -> None:
    b = _bundle("regression")
    X_tr, _, X_val, _ = train_val_split(b, seed=0)
    # no row of X_test appears in either split (test is held out entirely)
    train_rows = {r.tobytes() for r in np.vstack([X_tr, X_val])}
    assert not any(r.tobytes() in train_rows for r in b.X_test)
```

- [ ] **Step 3: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_splits.py -q`
Expected: FAIL (module missing).

- [ ] **Step 4: Implement** — `benchmarks/_common/splits.py`:

```python
"""Train/validation split helper for hyperparameter search (test untouched)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle


def train_val_split(
    bundle: DatasetBundle,
    *,
    frac: float = 0.2,
    seed: int,
    stratify: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split `bundle`'s train arrays into train/validation.

    :param frac: validation fraction of the train set.
    :param seed: deterministic split seed.
    :param stratify: stratify on `y`; defaults to True for binary classification.
    :returns: `(X_tr, y_tr, X_val, y_val)`. `bundle.X_test`/`y_test` are never read.
    """
    if stratify is None:
        stratify = bundle.task == "binary_classification"
    strat = bundle.y_train if stratify else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        bundle.X_train, bundle.y_train,
        test_size=frac, random_state=seed, stratify=strat,
    )
    return X_tr, y_tr, X_val, y_val
```

- [ ] **Step 5: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_splits.py -q`
Expected: PASS (4 passed). Also `uv run ruff check benchmarks/ tests/benchmarks/` and `uv run mypy benchmarks` clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock benchmarks/_common/splits.py tests/benchmarks/test_splits.py
git commit -m "$(cat <<'EOF'
bench: add optuna dep and train_val_split helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: per-flavor search space (`search_spaces.py`)

**Files:**
- Create: `benchmarks/_common/search_spaces.py`
- Test: `tests/benchmarks/test_search_spaces.py`

**Interfaces:**
- Consumes: `BenchmarkConfig`, `OptimizerSpec` (Phase 1); `optuna.Trial`.
- Produces: `suggest_config(trial, *, dataset, backend, mode, residual, epochs) -> BenchmarkConfig`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_search_spaces.py`:

```python
import pytest

optuna = pytest.importorskip("optuna")

from benchmarks._common.search_spaces import suggest_config


def _cfg(mode: str, residual: bool):
    study = optuna.create_study()
    trial = study.ask()
    return suggest_config(
        trial, dataset="syn", backend="torch", mode=mode, residual=residual, epochs=3,
    )


def test_absolute_searches_convex_fraction_within_unit_interval() -> None:
    cfg = _cfg("absolute", False)
    assert cfg.mode == "absolute" and cfg.residual is False
    assert 0.0 <= cfg.convex_fraction <= 1.0
    assert cfg.activation == "elu"
    assert cfg.epochs == 3
    assert 1 <= cfg.depth <= 4


def test_switch_uses_fixed_convex_fraction() -> None:
    # switch mode ignores convex_fraction; the sampler must NOT add it as a
    # search dimension (kept at the 0.5 default so studies don't carry a dead param).
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial, dataset="syn", backend="torch", mode="switch", residual=True, epochs=2,
    )
    assert cfg.mode == "switch" and cfg.residual is True
    assert cfg.convex_fraction == 0.5
    assert "convex_fraction" not in trial.params
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search_spaces.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — `benchmarks/_common/search_spaces.py`:

```python
"""Per-flavor Optuna search space producing a BenchmarkConfig."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec

if TYPE_CHECKING:
    import optuna


def suggest_config(
    trial: optuna.Trial,
    *,
    dataset: str,
    backend: Literal["torch", "jax", "keras"],
    mode: Literal["switch", "absolute"],
    residual: bool,
    epochs: int,
) -> BenchmarkConfig:
    """Sample a BenchmarkConfig for one (dataset, flavor) trial.

    `convex_fraction` is searched only for absolute mode; switch keeps 0.5.
    `activation` is fixed to "elu" in Phase 2a.
    """
    width = trial.suggest_categorical("width", [8, 16, 21, 32, 64])
    depth = trial.suggest_int("depth", 1, 4)
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.2)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr_decay = trial.suggest_float("lr_decay", 0.85, 1.0)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
    convex_fraction = (
        trial.suggest_float("convex_fraction", 0.0, 1.0) if mode == "absolute" else 0.5
    )
    task_metric = cast(
        'Literal["accuracy", "rmse", "mse"]',
        "accuracy" if dataset in _BINARY else "mse",
    )
    return BenchmarkConfig(
        dataset=dataset, backend=backend, mode=mode, residual=residual,
        depth=depth, width=int(width), activation="elu",
        convex_fraction=convex_fraction, embed_hidden=(int(width),),
        dropout=dropout,
        optimizer=OptimizerSpec("adam", lr, weight_decay),
        lr_decay=lr_decay, batch_size=int(batch_size), epochs=epochs,
        early_stopping=None, seeds=(0,),
        metrics=(task_metric,),
    )


# datasets whose primary metric is accuracy (binary classification)
_BINARY = {"compas", "heart", "loan"}
```

(The `suggest_config` keeps `seeds=(0,)` — search trials are single-seed for speed; the metric tuple matches the dataset's primary metric. `_BINARY` lists the binary paper datasets; regression datasets default to `mse`.)

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search_spaces.py -q`
Expected: PASS (2 passed). ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/search_spaces.py tests/benchmarks/test_search_spaces.py
git commit -m "$(cat <<'EOF'
bench: per-flavor Optuna search space (suggest_config)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: search engine + final eval (`search.py`)

**Files:**
- Create: `benchmarks/_common/search.py`
- Test: `tests/benchmarks/test_search.py`

**Interfaces:**
- Consumes: `train_val_split` (Task 1), `suggest_config` (Task 2), `run`/`BenchmarkConfig` and `aggregate` (Phase 1), `registry.load`, `DatasetBundle`, `optuna`.
- Produces: `flavor_name(mode, residual) -> str`; `StudyResult` dataclass; `search(bundle, *, mode, residual, backend, n_trials=50, seed=0, epochs=50, metric=None, storage=None) -> StudyResult`; `final_eval(bundle, best_params, *, mode, residual, backend, metric=None, seeds=range(10), epochs=50, top_k=5) -> Aggregate`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_search.py`:

```python
import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.search import StudyResult, final_eval, flavor_name, search


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5))
    y = (X[:, 0] + 0.1 * rng.normal(size=120)).astype(np.float64)
    return DatasetBundle(
        name="syn", task="regression",
        X_train=X, y_train=y, X_test=X[:30], y_test=y[:30],
        mono_increasing=(0,), mono_decreasing=(),
        feature_names=tuple(f"f{i}" for i in range(5)), metadata={},
    )


def test_flavor_name() -> None:
    assert flavor_name("switch", False) == "switch-plain"
    assert flavor_name("absolute", True) == "absolute-residual"


def test_search_two_trials_returns_finite_best() -> None:
    res = search(
        _bundle(), mode="switch", residual=False, backend="torch",
        n_trials=2, seed=0, epochs=1,
    )
    assert isinstance(res, StudyResult)
    assert res.n_trials == 2 and res.flavor == "switch-plain"
    assert np.isfinite(res.best_value)
    assert "lr" in res.best_params and "width" in res.best_params


def test_final_eval_returns_aggregate_on_test() -> None:
    b = _bundle()
    res = search(b, mode="switch", residual=False, backend="torch", n_trials=2, epochs=1)
    agg = final_eval(
        b, res.best_params, mode="switch", residual=False, backend="torch",
        seeds=range(2), epochs=1, top_k=2,
    )
    assert np.isfinite(agg.mean) and agg.n_selected == 2
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — `benchmarks/_common/search.py`:

```python
"""Optuna search engine over the Phase-1 run() harness (validation-driven)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.results import Aggregate, aggregate
from benchmarks._common.runner import run
from benchmarks._common.search_spaces import suggest_config
from benchmarks._common.splits import train_val_split

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle


def flavor_name(mode: str, residual: bool) -> str:
    return f"{mode}-{'residual' if residual else 'plain'}"


def _primary_metric(bundle: DatasetBundle) -> str:
    return "accuracy" if bundle.task == "binary_classification" else "mse"


def _lower_is_better(metric: str) -> bool:
    return metric in ("mse", "rmse")


@dataclass(frozen=True, slots=True)
class StudyResult:
    dataset: str
    flavor: str
    best_params: dict[str, Any]
    best_value: float
    n_trials: int


def _val_bundle(bundle: DatasetBundle, seed: int) -> DatasetBundle:
    """Throwaway bundle with the held-out validation set in the test slot.

    Lets the search reuse run() (which evaluates on X_test) to score on
    validation without ever touching the real test set.
    """
    X_tr, y_tr, X_val, y_val = train_val_split(bundle, seed=seed)
    return dataclasses.replace(
        bundle, X_train=X_tr, y_train=y_tr, X_test=X_val, y_test=y_val
    )


def search(
    bundle: DatasetBundle,
    *,
    mode: str,
    residual: bool,
    backend: str,
    n_trials: int = 50,
    seed: int = 0,
    epochs: int = 50,
    metric: str | None = None,
    storage: str | None = None,
) -> StudyResult:
    """Tune (dataset, flavor) HPs on a validation split via Optuna TPE."""
    metric = metric or _primary_metric(bundle)
    direction = "minimize" if _lower_is_better(metric) else "maximize"
    vb = _val_bundle(bundle, seed)

    def objective(trial: optuna.Trial) -> float:
        cfg: BenchmarkConfig = suggest_config(
            trial, dataset=bundle.name, backend=backend,  # type: ignore[arg-type]
            mode=mode, residual=residual, epochs=epochs,  # type: ignore[arg-type]
        )
        rows = run(cfg, vb)
        return float(rows[0].scores[metric])

    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=seed),
        storage=storage,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials)
    return StudyResult(
        dataset=bundle.name, flavor=flavor_name(mode, residual),
        best_params=dict(study.best_params), best_value=float(study.best_value),
        n_trials=len(study.trials),
    )


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
    top_k: int = 5,
) -> Aggregate:
    """Refit best HPs on the full train split; report TEST best-k-of-n."""
    metric = metric or _primary_metric(bundle)
    width = int(best_params["width"])
    cfg = BenchmarkConfig(
        dataset=bundle.name, backend=backend,  # type: ignore[arg-type]
        mode=mode, residual=residual,  # type: ignore[arg-type]
        depth=int(best_params["depth"]), width=width, activation="elu",
        convex_fraction=float(best_params.get("convex_fraction", 0.5)),
        embed_hidden=(width,), dropout=float(best_params["dropout"]),
        optimizer=OptimizerSpec(
            "adam", float(best_params["lr"]), float(best_params["weight_decay"])
        ),
        lr_decay=float(best_params["lr_decay"]), batch_size=int(best_params["batch_size"]),
        epochs=epochs, early_stopping=None, seeds=tuple(seeds),
        metrics=(metric,),  # type: ignore[arg-type]
    )
    rows = run(cfg, bundle)
    return aggregate(
        rows, metric=metric, lower_is_better=_lower_is_better(metric), top_k=top_k
    )
```

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_search.py -q`
Expected: PASS (3 passed). ruff + mypy clean. (Trials train a 1-epoch tiny model — fast.)

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/search.py tests/benchmarks/test_search.py
git commit -m "$(cat <<'EOF'
bench: Optuna search engine + final-eval over run() (validation-driven)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: search notebooks + results scaffold

**Files:**
- Create: `benchmarks/notebooks/search-{auto,heart,compas,loan,blog}.ipynb`
- Create: `benchmarks/results/phase2/.gitignore`
- Modify: `benchmarks/README.md` (manual-run instructions for the 20-study matrix)

**Interfaces:**
- Consumes: `search`, `final_eval`, `flavor_name`, `StudyResult` (Task 3); `registry.load`, `download.default_dest` (Phase 1).

- [ ] **Step 1: Create `benchmarks/results/phase2/.gitignore`**

```
*.db
*.jsonl
```

(Optuna SQLite study DBs and raw per-trial logs are ignored; the distilled
`<dataset>-<flavor>.json` files are committed by the maintainer after a run.)

- [ ] **Step 2: Create the five search notebooks** — `benchmarks/notebooks/search-<dataset>.ipynb`.

Each is a scaffold (valid JSON, **no executed outputs** — manual GPU runs produce them).
A markdown header cell (dataset + its monotone features + the budget choice), then a code
cell that runs the four flavors and writes per-flavor JSON. The `auto` notebook's code cell:

```python
import json
from pathlib import Path

from benchmarks.datasets.download import default_dest
from benchmarks.datasets.registry import load
from benchmarks._common.search import search, final_eval, flavor_name

DATASET = "auto"
BACKEND = "torch"
N_TRIALS = 50          # per (dataset, flavor); see header for per-dataset budget
EPOCHS = 50
OUT = Path("../results/phase2")
OUT.mkdir(parents=True, exist_ok=True)

bundle = load(DATASET, data_dir=default_dest())
flavors = [("switch", False), ("switch", True), ("absolute", False), ("absolute", True)]

for mode, residual in flavors:
    study = search(bundle, mode=mode, residual=residual, backend=BACKEND,
                   n_trials=N_TRIALS, epochs=EPOCHS,
                   storage=f"sqlite:///{OUT}/{DATASET}-{flavor_name(mode, residual)}.db")
    agg = final_eval(bundle, study.best_params, mode=mode, residual=residual,
                     backend=BACKEND, epochs=EPOCHS)
    rec = {
        "dataset": DATASET, "flavor": study.flavor,
        "best_params": study.best_params, "val_best": study.best_value,
        "test_metric": agg.metric, "test_mean": agg.mean, "test_std": agg.std,
        "n_seeds": agg.n_seeds, "n_selected": agg.n_selected,
    }
    (OUT / f"{DATASET}-{study.flavor}.json").write_text(json.dumps(rec, indent=2))
    print(study.flavor, agg.mean, "±", agg.std)
```

The other four notebooks are identical except `DATASET` (`heart`, `compas`, `loan`, `blog`)
and the header's per-dataset budget note (Loan/Blog: drop `N_TRIALS` to 25 and consider
fewer final-eval seeds — see README). `BACKEND` may be set to `jax`/`keras` for the
cross-backend view.

- [ ] **Step 3: Add a "Phase 2a: HP search" section to `benchmarks/README.md`**

Document: install `--group bench` (now includes optuna); run a search notebook in a
`gpu-*` devcontainer; it writes `results/phase2/<dataset>-<flavor>.json` (committed) and a
git-ignored `.db`; the 20-study matrix = 5 notebooks × 4 flavors; commit the JSON, then the
docs summary notebook renders them. Note the per-dataset budget guidance (open items).

- [ ] **Step 4: Verify the notebooks are valid JSON**

Run:
```bash
uv run python -c "import json,glob; [json.load(open(f)) for f in glob.glob('benchmarks/notebooks/*.ipynb')]; print('notebooks valid JSON')"
```
Expected: `notebooks valid JSON`.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/notebooks/ benchmarks/results/phase2/.gitignore benchmarks/README.md
git commit -m "$(cat <<'EOF'
bench: Phase 2a search notebooks + results scaffold

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: rendered flavor-comparison summary

**Files:**
- Create: `docs/benchmarks/flavor-comparison.ipynb`
- Modify: `docs/benchmarks/index.md` (toctree)

**Interfaces:**
- Consumes: committed `benchmarks/results/phase2/*.json` (Task 4 output, maintainer-populated).

- [ ] **Step 1: Create `docs/benchmarks/flavor-comparison.ipynb`** (scaffold, no executed outputs).

Markdown intro (what the flavor study compares; that numbers come from the maintainer's
search run), then a code cell that loads the committed JSON and renders a per-dataset table.
It must tolerate **missing** result files (renders an empty/partial table before the
maintainer's run, so the docs build is clean now):

```python
import json
from pathlib import Path

import pandas as pd

RESULTS = Path("../../benchmarks/results/phase2")
rows = []
for f in sorted(RESULTS.glob("*.json")) if RESULTS.exists() else []:
    rec = json.loads(f.read_text())
    rows.append({
        "dataset": rec["dataset"], "flavor": rec["flavor"],
        "test_metric": rec["test_metric"],
        "test_mean": round(rec["test_mean"], 4), "test_std": round(rec["test_std"], 4),
    })
df = pd.DataFrame(rows)
if df.empty:
    print("No Phase-2a results committed yet. Run benchmarks/notebooks/search-*.ipynb.")
else:
    display(df.pivot_table(index="dataset", columns="flavor", values="test_mean"))
df
```

- [ ] **Step 2: Wire into the docs toctree**

In `docs/benchmarks/index.md`, add `flavor-comparison` to the toctree (alongside the
existing `paper-reproduction/index`).

- [ ] **Step 3: Build the docs strict**

Run: `./tools/build-docs.sh`
Expected: exit 0 (the new notebook renders with the "no results yet" path; no executed
outputs required — `nb_execution_mode = "off"`).

- [ ] **Step 4: Commit**

```bash
git add docs/benchmarks/flavor-comparison.ipynb docs/benchmarks/index.md
git commit -m "$(cat <<'EOF'
docs: Phase 2a flavor-comparison summary notebook

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the executor

- **No real searches in execution.** Every task is unit-testable on synthetic data;
  the 20-study matrix and committed result JSON are manual maintainer runs (Task 4 README).
- **Backend gating:** sync the active backend (`uv sync --extra <backend> --group bench`)
  before running `test_search.py` so it doesn't skip; CI runs one backend per leg.
- **Median pruner:** intentionally omitted (see the Global Constraints deviation note) —
  TPE sampling only. Do not add a per-epoch pruning hook to `run()` in 2a.
- Do not commit Zenodo data, `dist/`, `*.db`, or `*.jsonl`.
