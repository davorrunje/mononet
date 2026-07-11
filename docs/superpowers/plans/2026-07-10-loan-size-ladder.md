# Loan size-ladder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the plumbing for a within-`loan` size-ladder experiment — a `subsample_train` helper, a run script that per-N tunes deep vs shallow `absolute` residual stacks and records Δ = IQM(deep) − IQM(shallow), a Δ-vs-N report/plot, a docs skeleton, and a RUNBOOK — mergeable with smoke results; the heavy full run is the GPU session's job.

**Architecture:** Benchmark-only (`benchmarks/`, out of the wheel). Reuse the existing `_common/search.py` `search`/`final_eval` primitives; the only new mechanism is stratified train-subsampling (fixed for the HP search, per-seed for the final multi-seed test so IQM absorbs subsample + init variance).

**Tech Stack:** Python 3.11+, PyTorch, optuna, scikit-learn, numpy, matplotlib (all in the `bench` group), pytest, Sphinx/myst-nb.

## Global Constraints

- Benchmark-only: **no** `mononet` package / kernel / `model_builder` change.
- Respect the protocol: **test set full and never touched** during search; selection on train-only CV; final numbers multi-seed on the held-out test via IQM.
- Line length 88 (ruff); strict mypy (`files = mononet, tests, benchmarks` — `docs/` not type-checked, but `benchmarks/` **is**).
- `DatasetBundle` is a `@dataclass(frozen=True, slots=True)` in `benchmarks/_common/bundle.py` with fields `name, task, X_train, y_train, X_test, y_test, mono_increasing, mono_decreasing, feature_names, metadata`. Mutate via `dataclasses.replace`.
- Reused signatures (do not change them): `search(bundle, *, mode, residual, backend, deep=False, n_trials=50, seed=0, epochs=50, n_jobs=1, n_splits=5, search_seeds=3, metric=None, storage=None) -> StudyResult` (`.best_params`, `.best_value`, `.flavor`); `final_eval(bundle, best_params, *, mode, residual, backend, metric=None, seeds=range(10), epochs=50) -> Aggregate` (`.metric,.mean,.std,.median,.iqm,.values`); `_count_collapses(values, *, task, base_rate, lower_is_better) -> int` (module-level in `search.py`). `loan` protocol budget: 25 trials, `n_splits=1`, `search_seeds=3`, 10 final seeds.
- Branch: `feat/loan-size-ladder` (already checked out). Bench group is installed here.
- Environment: gpu-torch container (torch installed). Do NOT use `--no-verify`; locale-error-only → prefix `LC_ALL=C.UTF-8 LANG=C.UTF-8`.

---

### Task 1: `subsample_train` helper + unit tests

**Files:**
- Modify: `benchmarks/_common/splits.py`
- Test: `tests/benchmarks/test_subsample.py` (new)

**Interfaces:**
- Produces: `subsample_train(bundle: DatasetBundle, n: int, *, seed: int, stratify: bool | None = None) -> DatasetBundle` — new bundle with `X_train`/`y_train` reduced to `n` stratified rows (class ratio preserved), deterministic per seed; test arrays and all other fields unchanged; `n >= len(train)` returns the bundle unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/benchmarks/test_subsample.py`:

```python
from __future__ import annotations

import dataclasses

import numpy as np

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.splits import subsample_train


def _bundle(n: int = 400, pos_frac: float = 0.3) -> DatasetBundle:
    rng = np.random.default_rng(0)
    y = (rng.random(n) < pos_frac).astype(np.float64)
    x = rng.standard_normal((n, 4))
    yt = (rng.random(80) < pos_frac).astype(np.float64)
    return DatasetBundle(
        name="synthetic",
        task="binary_classification",
        X_train=x,
        y_train=y,
        X_test=rng.standard_normal((80, 4)),
        y_test=yt,
        mono_increasing=(0, 1),
        mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"),
        metadata={},
    )


def test_subsample_size_and_test_untouched() -> None:
    """Subsample yields exactly n train rows and leaves test arrays identical."""
    b = _bundle()
    s = subsample_train(b, 100, seed=0)
    assert len(s.X_train) == 100
    assert len(s.y_train) == 100
    assert np.array_equal(s.X_test, b.X_test)
    assert np.array_equal(s.y_test, b.y_test)


def test_subsample_preserves_class_ratio() -> None:
    """Stratified subsample keeps the positive-class fraction within tolerance."""
    b = _bundle(pos_frac=0.3)
    s = subsample_train(b, 100, seed=0)
    assert abs(float(s.y_train.mean()) - float(b.y_train.mean())) < 0.05


def test_subsample_deterministic_and_seed_varies() -> None:
    """Same seed → identical rows; different seed → different rows."""
    b = _bundle()
    a0 = subsample_train(b, 100, seed=0)
    a0b = subsample_train(b, 100, seed=0)
    a1 = subsample_train(b, 100, seed=1)
    assert np.array_equal(a0.X_train, a0b.X_train)
    assert not np.array_equal(a0.X_train, a1.X_train)


def test_subsample_full_returns_unchanged() -> None:
    """n >= train size returns the same bundle object."""
    b = _bundle(n=400)
    assert subsample_train(b, 400, seed=0) is b
    assert subsample_train(b, 999, seed=0) is b
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/benchmarks/test_subsample.py -v`
Expected: FAIL — `subsample_train` does not exist (ImportError).

- [ ] **Step 3: Implement `subsample_train`**

In `benchmarks/_common/splits.py`, add `import dataclasses` at the top (after `from __future__ import annotations`), and append:

```python
def subsample_train(
    bundle: DatasetBundle,
    n: int,
    *,
    seed: int,
    stratify: bool | None = None,
) -> DatasetBundle:
    """Return a copy of `bundle` with its train split stratified-subsampled to `n`.

    The test arrays and all other fields are unchanged. Deterministic given
    `seed`. If `n >= len(bundle.X_train)`, the bundle is returned unchanged.

    :param n: Target number of train rows.
    :param seed: Deterministic subsample seed.
    :param stratify: Stratify on `y`; defaults to True for binary classification.
    :returns: A new `DatasetBundle` (or the original if `n` covers the whole train).
    """
    total = len(bundle.X_train)
    if n >= total:
        return bundle
    if stratify is None:
        stratify = bundle.task == "binary_classification"
    strat = bundle.y_train if stratify else None
    keep, _ = train_test_split(
        np.arange(total), train_size=n, random_state=seed, stratify=strat
    )
    return dataclasses.replace(
        bundle, X_train=bundle.X_train[keep], y_train=bundle.y_train[keep]
    )
```

`DatasetBundle` is only imported under `TYPE_CHECKING` in `splits.py`; that is fine — it is used solely in annotations and by `dataclasses.replace` at runtime (which needs the instance, not the class). No new runtime import required.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/benchmarks/test_subsample.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Lint + type-check + commit**

Run: `uv run ruff check benchmarks tests && uv run mypy`
Expected: clean.

```bash
git add benchmarks/_common/splits.py tests/benchmarks/test_subsample.py
git commit -m "bench: add stratified subsample_train helper (test split untouched)"
```

---

### Task 2: `loan_size_ladder_run.py` run script + smoke test

**Files:**
- Create: `benchmarks/loan_size_ladder_run.py`
- Test: `tests/benchmarks/test_loan_size_ladder.py` (new)

**Interfaces:**
- Consumes: `subsample_train` (Task 1); `search`, `final_eval`, `_count_collapses` from `benchmarks._common.search`.
- Produces: `run_ladder(bundle, *, ns, arms, backend="torch", n_trials, search_seeds, final_seeds, epochs) -> list[dict]` (one record per `(n, arm)`), and a `main()` that loads `loan` and writes `benchmarks/results/size-ladder/loan.json`. Each record: `{"n", "arm", "depth", "best_params", "cv_best", "test_metric", "test_mean", "test_std", "test_median", "test_iqm", "test_values", "n_collapse", "n_seeds"}`.

- [ ] **Step 1: Write the failing smoke test**

Create `tests/benchmarks/test_loan_size_ladder.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle  # noqa: E402
from benchmarks.loan_size_ladder_run import run_ladder  # noqa: E402


def _synthetic_bundle(n: int = 600) -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 4))
    # monotone-ish label: increasing in col 0/1, decreasing in col 2
    logit = x[:, 0] + x[:, 1] - x[:, 2]
    y = (logit > 0).astype(np.float64)
    xt = rng.standard_normal((120, 4))
    yt = ((xt[:, 0] + xt[:, 1] - xt[:, 2]) > 0).astype(np.float64)
    return DatasetBundle(
        name="synthetic",
        task="binary_classification",
        X_train=x,
        y_train=y,
        X_test=xt,
        y_test=yt,
        mono_increasing=(0, 1),
        mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"),
        metadata={},
    )


def test_run_ladder_smoke() -> None:
    """run_ladder returns one finite-IQM record per (n, arm) on a tiny bundle."""
    recs = run_ladder(
        _synthetic_bundle(),
        ns=(100, 400),
        arms=("shallow", "deep"),
        n_trials=2,
        search_seeds=1,
        final_seeds=range(2),
        epochs=1,
    )
    assert len(recs) == 4  # 2 ns x 2 arms
    for r in recs:
        assert r["n"] in (100, 400)
        assert r["arm"] in ("shallow", "deep")
        assert np.isfinite(r["test_iqm"])
        assert len(r["test_values"]) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_loan_size_ladder.py -v`
Expected: FAIL — `benchmarks.loan_size_ladder_run` does not exist.

- [ ] **Step 3: Implement the run script**

Create `benchmarks/loan_size_ladder_run.py`:

```python
"""Within-loan size-ladder: does deep monotone residual win with scale?

For each train size N and each arm (shallow D in [1,4] vs deep D in {6,10,16},
both absolute residual), tune HPs on an N-subsample, then refit + multi-seed
test on the full held-out test set (a fresh N-subsample per seed) and record the
IQM. See docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md.

Run: uv run --extra torch --group bench python -m benchmarks.loan_size_ladder_run
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._common.search import _count_collapses, final_eval, search
from benchmarks._common.splits import subsample_train

if TYPE_CHECKING:
    from collections.abc import Iterable

    from benchmarks._common.bundle import DatasetBundle

_NS: tuple[int, ...] = (5_000, 15_000, 45_000, 135_000, 1_000_000_000)  # last = full
_ARMS: tuple[str, ...] = ("shallow", "deep")


def _iqm(values: list[float]) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64))
    k = len(v) // 4
    return float(v[k : len(v) - k].mean()) if len(v) - 2 * k > 0 else float(v.mean())


def _ladder_eval(
    bundle: DatasetBundle,
    best_params: dict[str, Any],
    *,
    deep: bool,
    backend: str,
    n: int,
    final_seeds: Iterable[int],
    epochs: int,
) -> list[float]:
    """Per-seed: subsample train to n (seed s), refit, test on full test."""
    values: list[float] = []
    for s in final_seeds:
        b_s = subsample_train(bundle, n, seed=s)
        agg = final_eval(
            b_s,
            best_params,
            mode="absolute",
            residual=True,
            backend=backend,
            seeds=[s],
            epochs=epochs,
        )
        values.append(float(agg.values[0]))
    return values


def run_ladder(
    bundle: DatasetBundle,
    *,
    ns: tuple[int, ...] = _NS,
    arms: tuple[str, ...] = _ARMS,
    backend: str = "torch",
    n_trials: int = 25,
    search_seeds: int = 3,
    final_seeds: Iterable[int] = range(10),
    epochs: int = 50,
    n_splits: int = 1,
) -> list[dict[str, Any]]:
    """Run the size-ladder for `bundle`; return one record per (n, arm)."""
    seeds = list(final_seeds)
    base_rate = max(
        float(np.mean(bundle.y_test)), 1.0 - float(np.mean(bundle.y_test))
    )
    records: list[dict[str, Any]] = []
    for n in ns:
        for arm in arms:
            deep = arm == "deep"
            b_search = subsample_train(bundle, n, seed=0)
            study = search(
                b_search,
                mode="absolute",
                residual=True,
                deep=deep,
                backend=backend,
                n_trials=n_trials,
                epochs=epochs,
                n_splits=n_splits,
                search_seeds=search_seeds,
            )
            values = _ladder_eval(
                bundle,
                study.best_params,
                deep=deep,
                backend=backend,
                n=n,
                final_seeds=seeds,
                epochs=epochs,
            )
            n_eff = min(n, len(bundle.X_train))
            records.append(
                {
                    "n": n_eff,
                    "arm": arm,
                    "depth": int(study.best_params["depth"]),
                    "best_params": study.best_params,
                    "cv_best": study.best_value,
                    "test_metric": "accuracy",
                    "test_mean": float(np.mean(values)),
                    "test_std": float(np.std(values)),
                    "test_median": float(np.median(values)),
                    "test_iqm": _iqm(values),
                    "test_values": values,
                    "n_collapse": _count_collapses(
                        tuple(values),
                        task=bundle.task,
                        base_rate=base_rate,
                        lower_is_better=False,
                    ),
                    "n_seeds": len(values),
                }
            )
    return records


def main() -> None:
    """Load loan, run the full ladder, write the committed results JSON."""
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    bundle = load("loan", data_dir=default_dest())
    records = run_ladder(bundle)
    out = Path(__file__).resolve().parent / "results" / "size-ladder" / "loan.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"wrote {out} ({len(records)} records)")  # noqa: T201


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the smoke test passes**

Run: `uv run pytest tests/benchmarks/test_loan_size_ladder.py -v`
Expected: PASS (the `ns=(100, 400)` rungs both subsample the 600-row synthetic bundle; 2 arms × 2 ns = 4 records, each with 2 finite test values).

- [ ] **Step 5: Lint + type-check + commit**

Run: `uv run ruff check benchmarks tests && uv run mypy`
Expected: clean. (Note: `_count_collapses` and `final_eval`/`search` are imported from `benchmarks._common.search`; the leading-underscore import is intentional — a benchmark-internal reuse.)

```bash
git add benchmarks/loan_size_ladder_run.py tests/benchmarks/test_loan_size_ladder.py
git commit -m "bench: loan size-ladder run script (per-N deep-vs-shallow, per-seed subsample eval)"
```

---

### Task 3: Δ/plot report + docs skeleton + RUNBOOK + toctree

**Files:**
- Create: `benchmarks/_common/size_ladder_report.py`, `docs/benchmarks/loan-size-ladder.md`, `benchmarks/RUNBOOK-loan-ladder.md`
- Modify: `docs/benchmarks/index.md` (toctree + list), `docs/benchmarks/deep-residual-accuracy.md` (cross-link)
- Test: `tests/benchmarks/test_size_ladder_report.py` (new)

**Interfaces:**
- Consumes: the results-JSON schema from Task 2.
- Produces: `delta_by_n(records: list[dict]) -> list[dict]` — per N, `{"n", "deep_iqm", "shallow_iqm", "delta", "delta_lo", "delta_hi"}` (bootstrap percentile band over paired per-seed differences); `render_plot(records, out_path)` — writes a Δ-vs-N PNG (log-N x-axis).

- [ ] **Step 1: Write the failing report test**

Create `tests/benchmarks/test_size_ladder_report.py`:

```python
from __future__ import annotations

import pytest

from benchmarks._common.size_ladder_report import delta_by_n


def _rec(n: int, arm: str, iqm: float, values: list[float]) -> dict:
    return {"n": n, "arm": arm, "test_iqm": iqm, "test_values": values}


def test_delta_by_n_pairs_arms_and_signs_delta() -> None:
    """delta = deep_iqm - shallow_iqm per N, with a band bracketing it."""
    records = [
        _rec(100, "shallow", 0.60, [0.59, 0.61]),
        _rec(100, "deep", 0.58, [0.57, 0.59]),
        _rec(400, "shallow", 0.60, [0.60, 0.60]),
        _rec(400, "deep", 0.66, [0.65, 0.67]),
    ]
    rows = delta_by_n(records)
    by_n = {r["n"]: r for r in rows}
    assert by_n[100]["delta"] == pytest.approx(-0.02, abs=1e-9)
    assert by_n[400]["delta"] == pytest.approx(0.06, abs=1e-9)
    assert by_n[400]["delta_lo"] <= by_n[400]["delta"] <= by_n[400]["delta_hi"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/benchmarks/test_size_ladder_report.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the report helper**

Create `benchmarks/_common/size_ladder_report.py`:

```python
"""Δ = IQM(deep) − IQM(shallow) vs N from a size-ladder results JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

_BOOT = 2000
_BOOT_SEED = 0


def _iqm(values: np.ndarray) -> float:
    v = np.sort(values)
    k = len(v) // 4
    return float(v[k : len(v) - k].mean()) if len(v) - 2 * k > 0 else float(v.mean())


def delta_by_n(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per N: deep_iqm, shallow_iqm, delta, and a bootstrap percentile band.

    The band resamples the deep and shallow per-seed value vectors independently
    (`_BOOT` draws, seed `_BOOT_SEED`) and takes the 2.5/97.5 percentiles of the
    bootstrapped IQM difference.
    """
    rng = np.random.default_rng(_BOOT_SEED)
    by_n: dict[int, dict[str, dict[str, Any]]] = {}
    for r in records:
        by_n.setdefault(int(r["n"]), {})[r["arm"]] = r
    out: list[dict[str, Any]] = []
    for n in sorted(by_n):
        deep, shallow = by_n[n].get("deep"), by_n[n].get("shallow")
        if deep is None or shallow is None:
            continue
        dv = np.asarray(deep["test_values"], dtype=np.float64)
        sv = np.asarray(shallow["test_values"], dtype=np.float64)
        boot = np.array(
            [
                _iqm(rng.choice(dv, len(dv), replace=True))
                - _iqm(rng.choice(sv, len(sv), replace=True))
                for _ in range(_BOOT)
            ]
        )
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(
            {
                "n": n,
                "deep_iqm": float(deep["test_iqm"]),
                "shallow_iqm": float(shallow["test_iqm"]),
                "delta": float(deep["test_iqm"]) - float(shallow["test_iqm"]),
                "delta_lo": float(lo),
                "delta_hi": float(hi),
            }
        )
    return out


def render_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    """Render Δ-vs-N (log-N x-axis) with the bootstrap band to `out_path` (PNG)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = delta_by_n(records)
    ns = [r["n"] for r in rows]
    delta = [r["delta"] for r in rows]
    lo = [r["delta"] - r["delta_lo"] for r in rows]
    hi = [r["delta_hi"] - r["delta"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(0.0, color="0.7", lw=1)
    ax.errorbar(ns, delta, yerr=[lo, hi], marker="o", capsize=3)
    ax.set_xscale("log")
    ax.set_xlabel("train size N (log)")
    ax.set_ylabel("Δ IQM  (deep − shallow)")
    ax.set_title("loan: deep-vs-shallow accuracy gap vs training size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run to verify the report test passes**

Run: `uv run pytest tests/benchmarks/test_size_ladder_report.py -v`
Expected: PASS.

- [ ] **Step 5: Docs skeleton page**

Create `docs/benchmarks/loan-size-ladder.md`:

```markdown
# Loan size-ladder — does depth win with scale?

PR #72 found deep monotone residual stacks beat shallow ones **only** on
`loan`, the largest dataset. This experiment isolates the cause: it holds
`loan` fixed and varies the training-set size N, tuning a **deep** (`absolute`
residual, depth ∈ {6, 10, 16}) and a **shallow** (`absolute` residual, depth ∈
[1, 4]) arm independently at each N, then reports

$$\Delta(N) = \mathrm{IQM}_{\text{deep}}(N) - \mathrm{IQM}_{\text{shallow}}(N)$$

on the full held-out test set (10 seeds per arm; a fresh stratified
N-subsample per seed, so the IQM band captures subsample and training
variance). Method and protocol: {doc}`protocol` and the
[design spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md).

```{note}
Results are produced by the GPU session per `benchmarks/RUNBOOK-loan-ladder.md`;
this page is populated (plot + table) when that run lands.
```

<!-- The GPU run replaces this block with the Δ-vs-N plot and per-N table. -->
```

(No image reference yet — added by the GPU run — so strict `sphinx-build -W`
stays green.)

- [ ] **Step 6: RUNBOOK**

Create `benchmarks/RUNBOOK-loan-ladder.md` documenting the end-to-end GPU run: `uv run --extra torch-gpu --group bench python -m benchmarks.loan_size_ladder_run` → writes `benchmarks/results/size-ladder/loan.json`; then generate the plot (`python -c "import json; from benchmarks._common.size_ladder_report import render_plot; render_plot(json.load(open('benchmarks/results/size-ladder/loan.json')), Path('docs/_static/loan-size-ladder.png'))"`) and the per-N table (via `delta_by_n`), and fill `docs/benchmarks/loan-size-ladder.md` (embed the plot, paste the table, write the interpretation). Note expected cost: only the top 1–2 rungs are heavy; the full-N rung mirrors #72's loan run.

- [ ] **Step 7: Wire into the docs nav + cross-link**

In `docs/benchmarks/index.md`: add a bullet after the Deep-residual-accuracy bullet — `- [Loan size-ladder](loan-size-ladder.md) — does deep monotone residual win once the dataset is large enough?` — and add `loan-size-ladder` to the `{toctree}` list. In `docs/benchmarks/deep-residual-accuracy.md`, add one line near the loan finding cross-linking to `loan-size-ladder.md`.

- [ ] **Step 8: Build docs + full gate + commit**

Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html`
Expected: `build succeeded`, no warnings.
Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run pre-commit run --all-files`
Expected: all hooks Passed/Skipped.

```bash
git add benchmarks/_common/size_ladder_report.py tests/benchmarks/test_size_ladder_report.py \
        docs/benchmarks/loan-size-ladder.md benchmarks/RUNBOOK-loan-ladder.md \
        docs/benchmarks/index.md docs/benchmarks/deep-residual-accuracy.md
git commit -m "bench+docs: size-ladder Δ/plot report, docs skeleton, RUNBOOK, nav"
```

---

## Self-Review

**Spec coverage:**
- §3 experiment (isolate-depth absolute arms; per-N tuned option A; ladder; per-seed subsample; budget; Δ reporting) → Task 2 `run_ladder`/`_ladder_eval` + Task 3 `delta_by_n`. ✓
- §4 components: `subsample_train` → T1; run script + results schema → T2; Δ/plot helper → T3; docs page + RUNBOOK → T3. ✓
- §5 testing: subsample unit tests → T1; run-script smoke → T2; report test → T3; strict docs build + mypy → T3/global. ✓
- §6 scope split (plumbing + smoke here; full run via RUNBOOK) → docs skeleton has no fake numbers/plot; RUNBOOK owns the run. ✓
- §7 non-goals untouched (no other datasets/switch/mononet change). ✓

**Placeholder scan:** No TBD/TODO; every code step has full code; every command has expected output. The docs page intentionally defers plot/table to the GPU run (§6) rather than inventing numbers — a scope decision, not a placeholder.

**Type consistency:** `subsample_train(bundle, n, *, seed, stratify=None) -> DatasetBundle` defined in T1, consumed in T2 (`run_ladder`, `_ladder_eval`). Results-record keys written in T2 (`n, arm, test_iqm, test_values, ...`) are exactly the keys read in T3 (`delta_by_n` uses `n, arm, test_iqm, test_values`). `search`/`final_eval`/`_count_collapses` signatures match the reused definitions in Global Constraints. `_iqm` trimming (`k = len//4`) matches `make_tables._stats`.
