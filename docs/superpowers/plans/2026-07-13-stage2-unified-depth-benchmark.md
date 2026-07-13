# Stage-2 Unified Depth Benchmark — Implementation Plan (infra)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the benchmark the capability to run the unified two-stage depth study — ROC-AUC-primary scoring, a metric-aware significance gate, generator-backed synthetic datasets, all 15 datasets wired in, and a dual-GPU Stage-A launcher + a generalized size-ladder runner — so the actual Stage-A/B runs (controller-executed afterward) can produce the trustworthy post-#100 depth verdict.

**Architecture:** Benchmarks-only (no `mononet/` change). Extend the existing `benchmarks/_common` pipeline (`runner`, `search`, `screen_gate`) rather than fork it; add a generator dataset source to the registry; add two thin launchers. Much infra already exists from the #90 onboarding: `_BUDGET` already covers all 10 real datasets, and `run_dataset` already runs the 6 flavors — so this plan mostly adds the ROC-AUC metric, the gate analysis, the synthetic source, dataset wiring, and orchestration.

**Tech Stack:** Python 3.11+, PyTorch, Optuna, scikit-learn (1.9.0, already a `bench` dep), NumPy, typer, pytest, uv, ruff, strict mypy.

## Global Constraints

- Python 3.11+, ruff line length 88; strict mypy; MyST docstrings (`:param:`/`:returns:`/`:raises:`).
- No `mononet/` package change. Benchmarks + docs only. Pydantic banned (stdlib dataclasses).
- Heavy runs are **manual/controller-executed and committed with results**; CI never runs them. Every task here ships **fast, deterministic** unit/smoke tests only.
- Metric policy: classification primary = **ROC-AUC** (accuracy reported alongside); regression = MSE (`auto`) / RMSE (`blog`, synthetic). `_lower_is_better` stays `{mse, rmse}` (ROC-AUC is higher-better).
- Significance gate: Stage A emits **raw Δ + seed-bootstrap 95% CI** unconditionally; the **practical-margin floor is a parameter chosen post-results** (decision 2026-07-13) — do NOT hardcode a margin; tests pin gate *behavior*, not a margin value.
- Datasets: 10 real (`auto, heart, compas, loan, blog, adult, taiwan, polish, german, lc`) + 12 synthetic (`synth_<family>_c<level>`, family ∈ {additive, teacher_relu, teacher_elu, lattice}, level ∈ {low, mid, high}).
- Determinism: synthetic data is seeded; equivalence/committed artifacts never depend on live RNG.
- Never commit `*.db`/`*.jsonl` (Optuna storage is git-ignored); JSON results are the artifact.
- Commit proactively on the branch `feat/stage2-unified-depth-benchmark` (already created, holds the spec). Never commit to `main`.
- Spec: [`docs/superpowers/specs/2026-07-13-stage2-unified-depth-benchmark-design.md`](../specs/2026-07-13-stage2-unified-depth-benchmark-design.md). Read it before starting.

---

### Task 1: ROC-AUC as the classification primary metric

**Files:**
- Modify: `benchmarks/_common/config.py` (`metrics` Literal, ~line 80)
- Modify: `benchmarks/_common/runner.py` (`_score` loop, ~lines 266-281)
- Modify: `benchmarks/_common/search.py` (`_primary_metric`, ~line 41)
- Modify: `benchmarks/_common/search_spaces.py` (`suggest_config` `metric` Literal + `metrics` tuple)
- Test: `tests/benchmarks/test_runner.py`, `tests/benchmarks/test_search.py`

**Interfaces:**
- Consumes: `bundle.task`, `y_true`, `y_pred` (a [0,1]-ish score; classification `_predict` already returns a probability-scale output used with `>= 0.5`).
- Produces: `"roc_auc"` is a valid metric token computed as `sklearn.metrics.roc_auc_score(y_true, y_pred)`; `_primary_metric` returns `"roc_auc"` for `binary_classification`; classification configs carry `metrics=("roc_auc", "accuracy")` so accuracy is still reported.

- [ ] **Step 1: Write the failing test (runner scoring)**

Add to `tests/benchmarks/test_runner.py`:

```python
def test_score_computes_roc_auc_for_binary() -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from benchmarks._common.runner import _score_predictions  # thin, see Step 3

    y_true = np.array([0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([0.1, 0.4, 0.35, 0.9])  # one swap -> AUC 0.75
    scores = _score_predictions(y_pred, y_true, binary=True, metrics=("roc_auc", "accuracy"))
    assert scores["roc_auc"] == pytest.approx(roc_auc_score(y_true, y_pred))
    assert "accuracy" in scores
```

(If `runner._score` is not import-friendly as-is, extract the metric loop into a pure
`_score_predictions(y_pred, y_true, *, binary, metrics)` helper in Step 3 and have `_score` call it.)

- [ ] **Step 2: Run it — fails**

Run: `uv run pytest tests/benchmarks/test_runner.py::test_score_computes_roc_auc_for_binary -q`
Expected: FAIL (`roc_auc` unknown / helper missing).

- [ ] **Step 3: Implement**

`benchmarks/_common/config.py`: widen the metrics Literal:
```python
    metrics: tuple[Literal["accuracy", "rmse", "mse", "roc_auc"], ...]
```
`benchmarks/_common/runner.py`: extract the metric loop into a pure helper and add the branch:
```python
def _score_predictions(
    y_pred: np.ndarray,  # type: ignore[type-arg]
    y_true: np.ndarray,  # type: ignore[type-arg]
    *,
    binary: bool,
    metrics: tuple[str, ...],
) -> dict[str, float]:
    """Compute the requested metrics from predictions and targets."""
    scores: dict[str, float] = {}
    mse_val: float | None = None
    for metric in metrics:
        if metric == "mse":
            mse_val = float(np.mean((y_pred - y_true) ** 2))
            scores["mse"] = mse_val
        elif metric == "rmse":
            if mse_val is None:
                mse_val = float(np.mean((y_pred - y_true) ** 2))
            scores["rmse"] = math.sqrt(mse_val)
        elif metric == "accuracy":
            if not binary:
                raise ValueError("accuracy metric requires binary_classification task")
            scores["accuracy"] = float(np.mean((y_pred >= 0.5).astype(np.float64) == y_true))
        elif metric == "roc_auc":
            if not binary:
                raise ValueError("roc_auc metric requires binary_classification task")
            from sklearn.metrics import roc_auc_score

            scores["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        else:
            raise ValueError(f"Unknown metric: {metric!r}")
    return scores
```
Replace the inline loop in `_score` with `return _score_predictions(y_pred, y_true, binary=binary, metrics=cfg.metrics)`.

`benchmarks/_common/search.py`:
```python
def _primary_metric(bundle: DatasetBundle) -> str:
    return "roc_auc" if bundle.task == "binary_classification" else "mse"
```
`benchmarks/_common/search_spaces.py`: add `"roc_auc"` to `suggest_config`'s `metric` Literal, and when the primary is `roc_auc`, set the config to report accuracy too:
```python
    metrics = ("roc_auc", "accuracy") if metric == "roc_auc" else (metric,)
    return BenchmarkConfig(..., metrics=metrics, ...)
```
(Replace the existing `metrics=(metric,)`.)

- [ ] **Step 4: Add the `_primary_metric` test + run**

Add to `tests/benchmarks/test_search.py` a test that `_primary_metric` returns `"roc_auc"` for a binary bundle and `"mse"` for a regression bundle (build tiny `DatasetBundle`s or reuse existing fixtures). Then:
Run: `uv run pytest tests/benchmarks/test_runner.py tests/benchmarks/test_search.py -q`
Expected: PASS.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check benchmarks/_common tests/benchmarks && uv run mypy benchmarks
git add benchmarks/_common/config.py benchmarks/_common/runner.py benchmarks/_common/search.py \
        benchmarks/_common/search_spaces.py tests/benchmarks/test_runner.py tests/benchmarks/test_search.py
git commit -m "bench: ROC-AUC primary metric for classification (report accuracy alongside)"
```

---

### Task 2: Metric-aware significance gate + Δ/CI extractor

**Files:**
- Modify: `benchmarks/_common/screen_gate.py` (keep `gate`; make its use metric-agnostic + documented)
- Create: `benchmarks/_common/stage2_gate.py` (Δ + seed-bootstrap CI over result JSONs → verdict)
- Test: `tests/benchmarks/test_stage2_gate.py` (NEW)

**Interfaces:**
- Consumes: per-flavor Stage-A result JSONs for one dataset (schema = `run_dataset` output: per-seed test scores under the primary metric), the primary metric name, and `_lower_is_better`.
- Produces: `stage2_gate.dataset_delta(results, metric) -> DeltaResult(delta_point, delta_lo, delta_hi, best_shallow_flavor, best_deep_flavor)` where Δ is **signed so positive = deep better** (sign-normalized by metric direction); and `stage2_gate.verdict(delta: DeltaResult, margin: float) -> Literal["deep-better","neutral","deep-worse"]` layered on `screen_gate.gate`.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_stage2_gate.py`:

```python
from __future__ import annotations

import pytest

from benchmarks._common.stage2_gate import DeltaResult, verdict


@pytest.mark.parametrize(
    ("lo", "point", "margin", "expect"),
    [
        (0.01, 0.03, 0.02, "deep-better"),   # significant + clears margin
        (0.01, 0.015, 0.02, "neutral"),      # significant but below margin
        (-0.01, 0.03, 0.02, "neutral"),      # CI touches 0
        (-0.05, -0.03, 0.02, "deep-worse"),  # significantly worse
    ],
)
def test_verdict(lo: float, point: float, margin: float, expect: str) -> None:
    d = DeltaResult(delta_point=point, delta_lo=lo, delta_hi=point + 0.01,
                    best_shallow_flavor="absolute-plain", best_deep_flavor="absolute-deep")
    assert verdict(d, margin=margin) == expect


def test_delta_sign_is_normalized_for_lower_is_better() -> None:
    # deep MSE 0.10 vs shallow 0.15 -> deep BETTER -> positive delta
    from benchmarks._common.stage2_gate import _signed_improvement
    assert _signed_improvement(deep=0.10, shallow=0.15, lower_is_better=True) == pytest.approx(0.05)
    # deep AUC 0.80 vs shallow 0.75 -> deep BETTER -> positive delta
    assert _signed_improvement(deep=0.80, shallow=0.75, lower_is_better=False) == pytest.approx(0.05)
```

- [ ] **Step 2: Run it — fails**

Run: `uv run pytest tests/benchmarks/test_stage2_gate.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `stage2_gate.py`**

Reuse the existing bootstrap/IQM utilities (`benchmarks/_common/results.py` provides `interquartile_mean`; the loan-ladder / screen already bootstrap Δ — reuse that helper, do not reinvent). Implement:

```python
"""Stage-2 deep-vs-shallow gate: signed Δ + seed-bootstrap CI + verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from benchmarks._common.screen_gate import gate


@dataclass(frozen=True, slots=True)
class DeltaResult:
    delta_point: float
    delta_lo: float
    delta_hi: float
    best_shallow_flavor: str
    best_deep_flavor: str


def _signed_improvement(*, deep: float, shallow: float, lower_is_better: bool) -> float:
    """Δ normalized so positive == deep is better, regardless of metric direction."""
    return (shallow - deep) if lower_is_better else (deep - shallow)


def verdict(delta: DeltaResult, *, margin: float) -> Literal["deep-better", "neutral", "deep-worse"]:
    """Classify a dataset's deep-vs-shallow gap given a chosen practical margin.

    :param delta: Signed Δ (positive == deep better) with its 95% CI.
    :param margin: Practical-significance floor (chosen post-results).
    :returns: ``deep-better`` when significantly and practically positive;
        ``deep-worse`` when significantly negative; else ``neutral``.
    """
    if gate(delta.delta_lo, delta.delta_point, margin) == "ladder":
        return "deep-better"
    if delta.delta_hi < 0.0 and abs(delta.delta_point) >= margin:
        return "deep-worse"
    return "neutral"
```
Add `dataset_delta(result_dir, dataset, metric, *, n_boot=..., seed=...)` that: loads the 6 flavor JSONs, selects best-shallow (of 4 non-deep) and best-deep (of 2) by the primary metric's point estimate, and seed-bootstraps `_signed_improvement` over the per-seed test scores to get `(delta_point, delta_lo, delta_hi)`. Reuse the existing bootstrap util; if the loan-ladder's bootstrap isn't importable as a function, extract it into `results.py` in this task and have both call it (note the extraction in the report).

- [ ] **Step 4: Run + a small real-JSON smoke**

Run: `uv run pytest tests/benchmarks/test_stage2_gate.py -q` → PASS.
Add one smoke test that `dataset_delta` runs on a tiny hand-written pair of flavor JSONs in `tmp_path` (2 seeds each) and returns a finite Δ with `lo <= point <= hi`.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check benchmarks/_common tests/benchmarks && uv run mypy benchmarks
git add benchmarks/_common/stage2_gate.py benchmarks/_common/screen_gate.py \
        benchmarks/_common/results.py tests/benchmarks/test_stage2_gate.py
git commit -m "bench: metric-aware Stage-2 deep-vs-shallow gate (signed Δ + bootstrap CI)"
```

---

### Task 3: Synthetic generator + generator-backed registry source

**Files:**
- Create: `benchmarks/datasets/synthetic.py` (port from `origin/feat/monotone-depth-probe`)
- Modify: `benchmarks/datasets/spec.py` (generator source variant)
- Modify: `benchmarks/datasets/registry.py` (dispatch generator vs CSV)
- Modify: `benchmarks/_common/search.py` (`_BUDGET` += synthetic preset)
- Test: `tests/benchmarks/test_synthetic.py` (NEW), extend `tests/benchmarks/test_registry.py` if present

**Interfaces:**
- Consumes: `DatasetBundle`, a seed, `(family, c, d, n_train, n_test)`.
- Produces: `synthetic.synth_monotone(kind, c, ...)` → `DatasetBundle` (all features monotone-increasing, regression). Registry `load("synth_teacher_relu_cmid", data_dir=...)` returns the generated bundle (ignoring `data_dir`); `DATASETS` includes 12 `synth_*` keys. `teacher_relu` and `teacher_elu` produce **different** targets.

- [ ] **Step 1: Port the generator + write the failing test**

Port the file verbatim (it is correct on the probe branch — `_teacher` applies its `act`, so `teacher_relu != teacher_elu`):
```bash
git show origin/feat/monotone-depth-probe:benchmarks/datasets/synthetic.py > benchmarks/datasets/synthetic.py
```
Create `tests/benchmarks/test_synthetic.py`:
```python
from __future__ import annotations

import numpy as np

from benchmarks.datasets.synthetic import synth_monotone


def test_teacher_relu_and_elu_differ() -> None:
    r = synth_monotone("teacher_relu", c=2, d=6, n_train=256, n_test=64, seed=0)
    e = synth_monotone("teacher_elu", c=2, d=6, n_train=256, n_test=64, seed=0)
    assert float(np.max(np.abs(r.y_train - e.y_train))) > 1e-6  # activation switch is wired


def test_targets_are_monotone_increasing() -> None:
    b = synth_monotone("teacher_relu", c=2, d=6, n_train=512, n_test=64, seed=0)
    x = b.X_train.copy()
    base = _predict_target("teacher_relu", 2, 6, 0, x)
    x[:, 0] += 0.5  # raise one increasing feature
    assert np.all(_predict_target("teacher_relu", 2, 6, 0, x) - base >= -1e-8)


def test_deterministic() -> None:
    a = synth_monotone("lattice", c=2, d=6, n_train=128, n_test=32, seed=1)
    b = synth_monotone("lattice", c=2, d=6, n_train=128, n_test=32, seed=1)
    assert np.array_equal(a.y_train, b.y_train)
```
(`_predict_target` — a small local helper re-running the seeded target fn on modified `x`; or assert monotonicity by sorting one feature and checking the target is non-decreasing along it. Use whichever the ported module exposes; if it exposes only `synth_monotone`, test monotonicity by perturbing a feature up and re-generating with the same seed is not valid — instead evaluate the committed target fn. Simplest: import `_target_fn` and evaluate it directly on `x` and `x`-perturbed.)

- [ ] **Step 2: Run — fails**

Run: `uv run pytest tests/benchmarks/test_synthetic.py -q`
Expected: FAIL (registry not wired / import path) — or PASS for the generator-only tests if the port is clean; the registry test (Step 4) is the one that must fail first.

- [ ] **Step 3: Add a generator source to the registry**

In `benchmarks/datasets/spec.py`, add an optional generator field to `DatasetSpec` (or a sibling `GeneratedSpec`) carrying `kind`, `c`, `d`, `n_train`, `n_test`, `seed`; a CSV spec leaves it `None`. In `benchmarks/datasets/registry.py::load`, dispatch: if the spec is generator-backed, call `synth_monotone(...)` and return the bundle (ignore `data_dir`); else the existing CSV path. Register the 12 `synth_<family>_c<low|mid|high>` entries in `DATASETS_SPEC` with concrete `c` levels per family (pin: e.g. `low=1, mid=2, high=4` teacher depth / lattice levels; `d=6`, `n_train=16000`, `n_test=4000`, distinct seeds). Keep the mono directions all-increasing.

- [ ] **Step 4: Registry test + budgets**

Add to the registry test: `load("synth_teacher_relu_cmid", data_dir=<any>)` returns a monotone regression `DatasetBundle` of the expected shape. Add a synthetic budget to `benchmarks/_common/search.py::_BUDGET` (cheap: e.g. `(25, range(5), 1)`) applied to every `synth_*` key (a prefix rule in `run_dataset`'s `_BUDGET.get`, or explicit entries).
Run: `uv run pytest tests/benchmarks/test_synthetic.py tests/benchmarks/test_registry.py -q` → PASS.

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check benchmarks tests/benchmarks && uv run mypy benchmarks
git add benchmarks/datasets/synthetic.py benchmarks/datasets/spec.py benchmarks/datasets/registry.py \
        benchmarks/_common/search.py tests/benchmarks/test_synthetic.py tests/benchmarks/test_registry.py
git commit -m "bench: generator-backed synthetic datasets in the registry (probe folded in)"
```

---

### Task 4: Wire all 15 datasets into the benchmark CLI

**Files:**
- Modify: `benchmarks/search.py` (`_ALL_DATASETS`)
- Test: `tests/benchmarks/test_search_cli.py`

**Interfaces:**
- Consumes: the registry keys from Tasks 3 (synthetic) and the already-present real datasets.
- Produces: `search.py` default `--datasets` covers all 10 real + 12 synthetic; `--dry-run` lists them.

- [ ] **Step 1: Failing test**

Add to `tests/benchmarks/test_search_cli.py` a test invoking the typer app with `--dry-run` (no `--datasets`) and asserting the printed plan contains all 10 real names + at least the 12 `synth_*` names (use typer's `CliRunner`). Expected FAIL (only 5 present).

- [ ] **Step 2: Run — fails.** `uv run pytest tests/benchmarks/test_search_cli.py -q`

- [ ] **Step 3: Implement**

```python
_ALL_DATASETS = [
    "auto", "heart", "compas", "loan", "blog",
    "adult", "taiwan", "polish", "german", "lc",
    *[f"synth_{fam}_c{lvl}"
      for fam in ("additive", "teacher_relu", "teacher_elu", "lattice")
      for lvl in ("low", "mid", "high")],
]
```
(Match the exact synthetic keys registered in Task 3.)

- [ ] **Step 4: Run — passes.** `uv run pytest tests/benchmarks/test_search_cli.py -q`

- [ ] **Step 5: Smoke the whole plan (no training)**

Run: `uv run --extra torch --group bench python -m benchmarks.search --dry-run`
Expected: prints all 22 datasets × 6 flavors, exits 0.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/search.py tests/benchmarks/test_search_cli.py
git commit -m "bench: wire all 10 real + 12 synthetic datasets into the benchmark CLI"
```

---

### Task 5: Dual-GPU Stage-A launcher

**Files:**
- Create: `benchmarks/stage2_launch.py`
- Test: `tests/benchmarks/test_stage2_launch.py` (NEW)

**Interfaces:**
- Consumes: the dataset list, a device pool, `benchmarks.search.run_dataset` (via subprocess `python -m benchmarks.search --datasets <one> ...`).
- Produces: `main(datasets, devices, out_dir, storage_dir)` runs one dataset per subprocess pinned to a device (`MONONET_TORCH_DEVICE`), round-robin over the pool, each with `--n-jobs 1`; mirrors `benchmarks/screen_launch.py`.

- [ ] **Step 1: Failing test**

Model on `tests/benchmarks/test_screen_launch.py`. Create `tests/benchmarks/test_stage2_launch.py` asserting: (a) with `devices=["cuda:0","cuda:1"]` and 4 datasets, the launcher assigns them round-robin (monkeypatch the subprocess runner to record `(dataset, device)` pairs); (b) each subprocess command includes `--n-jobs 1` (or `--n-jobs`, `1`). Expected FAIL (module missing).

- [ ] **Step 2: Run — fails.** `uv run pytest tests/benchmarks/test_stage2_launch.py -q`

- [ ] **Step 3: Implement `stage2_launch.py`**

Copy the device-pool structure from `benchmarks/screen_launch.py` (a `Queue` of devices, one worker per pool slot, `MONONET_TORCH_DEVICE` in the subprocess env), but invoke `python -m benchmarks.search --datasets <name> --n-jobs 1 --storage-dir <dir> --out-dir <dir>` per dataset instead of `large_screen_run`. Keep the `# noqa: T201` prints and the merge-of-per-dataset behavior (here each subprocess already writes its own flavor JSONs, so no merge needed — just wait for all). `n_jobs` inside each search stays 1 (threaded-Optuna deadlock).

- [ ] **Step 4: Run — passes.** `uv run pytest tests/benchmarks/test_stage2_launch.py -q`

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check benchmarks/stage2_launch.py tests/benchmarks/test_stage2_launch.py && uv run mypy benchmarks/stage2_launch.py
git add benchmarks/stage2_launch.py tests/benchmarks/test_stage2_launch.py
git commit -m "bench: dual-GPU Stage-A launcher (one dataset per process, n_jobs=1)"
```

---

### Task 6: Generalized size-ladder runner (Stage B)

**Files:**
- Modify: `benchmarks/loan_size_ladder_run.py` → generalize (or create `benchmarks/size_ladder_run.py`)
- Modify: `benchmarks/loan_ladder_launch.py` → accept `--dataset`
- Test: `tests/benchmarks/test_loan_ladder_launch.py` (extend) / new `test_size_ladder.py`

**Interfaces:**
- Consumes: any large dataset name (not just `loan`), the ladder sizes, `run_dataset`/`search`+`final_eval`.
- Produces: `size_ladder_run.ladder(dataset, sizes, ...)` runs deep-vs-shallow tuned independently at each subsample size for the named dataset; launcher `--dataset <name>` distributes sizes/arms across GPUs.

- [ ] **Step 1: Failing test**

Extend the loan-ladder launch test (or new `test_size_ladder.py`) to assert the runner/launcher accept a `--dataset` other than `loan` (e.g. `lc`) and thread it through to `run_dataset`/subsampling. Model on the existing `tests/benchmarks/test_loan_ladder_launch.py`. Expected FAIL (hardcoded `loan`).

- [ ] **Step 2: Run — fails.**

- [ ] **Step 3: Implement**

Replace the hardcoded `"loan"` in `loan_size_ladder_run.py` / `loan_ladder_launch.py` with a `dataset` parameter (default `"loan"` for back-compat) threaded into the subsample + `run_dataset` calls; keep the size-ladder subsampling, per-size independent tuning, IQM + bootstrap Δ(size) logic unchanged. Rename to `size_ladder_run.py` only if it doesn't churn imports (else keep filename, add the param). Guard: only large datasets (`n_train ≥ 20_000`) are valid; raise a clear error otherwise.

- [ ] **Step 4: Run — passes.**

- [ ] **Step 5: Lint/type + commit**

```bash
uv run ruff check benchmarks tests/benchmarks && uv run mypy benchmarks
git add benchmarks/loan_size_ladder_run.py benchmarks/loan_ladder_launch.py tests/benchmarks/
git commit -m "bench: generalize the size-ladder runner to any large dataset (Stage B)"
```

---

## Post-plan (controller-executed, after this infra lands)

Not code tasks — run on the dual-GPU box, committed with results, then a follow-up PR:

1. **Stage A run:** `benchmarks.stage2_launch` across `cuda:0,cuda:1` over all 22 datasets (6 flavors). Commit `benchmarks/results/stage2/*.json`. (Full sweep is many GPU-hours; monitor, resume via Optuna `.db`.)
2. **Gate analysis:** compute `stage2_gate.dataset_delta` for every dataset; **choose the practical-margin floor(s) from the observed Δ distribution** (deferred decision); record per-dataset verdicts.
3. **Stage B run:** size-ladder the large real datasets with a `deep-better` verdict (if any). If none gate, document "no dataset advanced to a ladder."
4. **Docs + PR:** fill `docs/benchmarks/deep-residual-accuracy.md` (unified 10-real + 12-synthetic table, ROC-AUC primary + accuracy, gate verdicts, synthetic depth-vs-complexity reading); fill the before/after `{note}` in `docs/concepts/monotonic-residual.md`; add ladder page(s) for gated datasets; refresh/replace the README table (decide full-table vs summary+link); retire `docs/benchmarks/large-dataset-screen.md`. Open the follow-up PR; **close #90 (superseded)** and **#99 (folded in)**.

## Self-Review notes

- **Spec coverage:** ROC-AUC metric (T1) ✓; metric-aware gate + deferred margin (T2) ✓; synthetic generator + registry source + teacher_elu guard (T3) ✓; 15 datasets wired (T4) ✓; dual-GPU launcher (T5) ✓; generalized size-ladder (T6) ✓; Stage A/B runs + docs + #90/#99 closure = controller post-plan ✓; "no assumption deep wins" honored (gate is symmetric: deep-better / neutral / deep-worse) ✓.
- **Already-present infra (not re-built):** `_BUDGET` covers all 10 real datasets; `run_dataset` runs 6 flavors; `screen_gate.gate` exists (reused). Verified against `benchmarks/_common/search.py`.
- **Type consistency:** `_score_predictions(y_pred, y_true, *, binary, metrics)`; `DeltaResult`/`verdict(delta, *, margin)`/`_signed_improvement(*, deep, shallow, lower_is_better)`; `synth_monotone(kind, c, *, d, n_train, n_test, seed)`; synthetic keys `synth_<family>_c<low|mid|high>` identical in T3 (registry) and T4 (CLI list).
- **Deferred to plan-time decisions, now pinned:** synthetic `c` = {low:1, mid:2, high:4}, `d=6`, `n_train=16000`; synthetic budget `(25, range(5), 1)`; margin(s) intentionally NOT pinned (post-results). README table shape decided in post-plan step 4.
