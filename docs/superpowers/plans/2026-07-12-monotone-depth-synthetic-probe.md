# Monotone-depth synthetic probe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the synthetic-monotone-target probe from the approved spec so a GPU run can produce Δ(c) — whether monotone depth ever helps as target complexity grows — separating expressivity/data-simplicity from optimization.

**Architecture:** A synthetic-target generator emits `DatasetBundle`s (regression, all features monotone-increasing) with a complexity knob `c`; a probe runner sweeps `(kind, c)` through the existing `search` + `final_eval` deep/shallow arms (MSE, task-aware) and records per-arm IQM; a report computes Δ(c) = MSE(shallow) − MSE(deep) with a seed-bootstrap band (positive ⇒ depth helps) and renders the headline plot. Reuses the benchmark harness; no `mononet` change.

**Tech Stack:** Python 3.11+, numpy, `mononet.core.reference` (teacher), Optuna/PyTorch (via `search`/`final_eval`), matplotlib, Sphinx, uv.

## Global Constraints

- Python 3.11+, ruff line length 88; strict mypy; MyST field-list docstrings (no `:type:`/`:rtype:`); stdlib dataclasses.
- Benchmark-only: NO `mononet/` package/kernel change (the teacher *calls* `mononet.core.reference`, adds nothing to it). `benchmarks/` stays out of the wheel.
- All targets **monotone non-decreasing in every input**, domain `[0,1]^d`, `d = 6`; monotonicity asserted numerically at generation.
- Task = **regression**, metric = **MSE**; noise-free targets (approximation test). Δ sign convention: **Δ = MSE(shallow) − MSE(deep)** so **positive ⇒ depth helps**.
- Student arms identical to the screen: `absolute` `MonoResidual`, deep depth ∈ {6,10,16} vs shallow ∈ [1,4], same `search_spaces`.
- Families (per spec §8): **additive** (control), **teacher** (monotone-MLP depth sweep), **lattice** (max/min nesting). Interaction-order polynomial deferred.

## File structure

- Create `benchmarks/datasets/synthetic.py` — `synth_monotone(...) -> DatasetBundle` + the three target generators + numerical-monotonicity check.
- Create `benchmarks/monotone_depth_probe_run.py` — `probe_dataset(...)` + CLI sweeping `(kind, c)`.
- Create `benchmarks/_common/depth_probe_report.py` — `delta_by_c(...)` (MSE-aware) + `render_probe_plot(...)`.
- Create `docs/benchmarks/monotone-depth-probe.md`, `benchmarks/RUNBOOK-depth-probe.md`; modify `docs/benchmarks/index.md` (toctree).
- Tests under `tests/benchmarks/`.

---

### Task 1: Synthetic monotone-target generator

**Files:**
- Create: `benchmarks/datasets/synthetic.py`
- Test: `tests/benchmarks/test_synthetic.py`

**Interfaces:**
- Produces `synth_monotone(kind: Literal["additive","teacher","lattice"], c: int, *, d: int = 6, n_train: int = 4000, n_test: int = 2000, seed: int = 0) -> DatasetBundle` — regression bundle, `X ~ U[0,1]^d`, `y = f(X)` standardized to unit variance, `feature_names = ("x0",…)`, all indices in `mono_increasing`, `mono_decreasing = ()`, `task="regression"`, `metadata={"kind":kind,"c":str(c)}`. Task 2 consumes it.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_synthetic.py
from __future__ import annotations

import numpy as np
import pytest

from benchmarks.datasets.synthetic import synth_monotone


@pytest.mark.parametrize("kind", ["additive", "teacher", "lattice"])
def test_synth_is_monotone_and_shaped(kind: str) -> None:
    b = synth_monotone(kind, c=4, d=6, n_train=500, n_test=200, seed=0)
    assert b.task == "regression"
    assert b.X_train.shape == (500, 6) and b.X_test.shape == (200, 6)
    assert b.mono_increasing == (0, 1, 2, 3, 4, 5) and b.mono_decreasing == ()
    assert np.isfinite(b.y_train).all()
    # numerical monotonicity: raising any single feature never lowers f
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, size=(64, 6))
    from benchmarks.datasets.synthetic import _target_fn
    f = _target_fn(kind, c=4, d=6, seed=0)
    base = f(x)
    for j in range(6):
        xp = x.copy()
        xp[:, j] = np.minimum(1.0, x[:, j] + 0.1)
        assert (f(xp) - base >= -1e-9).all(), f"non-monotone in dim {j}"


def test_synth_deterministic_per_seed() -> None:
    a = synth_monotone("teacher", c=4, seed=0)
    b = synth_monotone("teacher", c=4, seed=0)
    assert np.array_equal(a.y_train, b.y_train)
    assert not np.array_equal(a.y_train, synth_monotone("teacher", c=4, seed=1).y_train)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_synthetic.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `synthetic.py`**

```python
"""Synthetic monotone-regression targets with a complexity knob (depth probe).

Every target is non-decreasing in every input on ``[0,1]^d`` (asserted
numerically). Three families: ``additive`` (control), ``teacher`` (a seeded
monotone MLP of depth ``c`` — non-negative weights + a monotone activation, so
the composite is monotone), ``lattice`` (nested element-wise max/min of monotone
terms, nesting depth ``c`` — piecewise complexity without oscillation).
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
import numpy.typing as npt

from benchmarks._common.bundle import DatasetBundle

_Kind = Literal["additive", "teacher", "lattice"]
Array = npt.NDArray[np.floating]


def _softplus(z: Array) -> Array:
    return np.logaddexp(0.0, z)  # monotone increasing


def _additive(d: int, seed: int) -> Callable[[Array], Array]:
    rng = np.random.default_rng(seed)
    # per-feature monotone piecewise-linear via cumulative non-negative slopes
    knots = np.sort(rng.uniform(0, 1, size=(d, 5)), axis=1)
    slopes = rng.uniform(0.2, 1.0, size=(d, 6))

    def f(x: Array) -> Array:
        out = np.zeros(len(x))
        for j in range(d):
            xs = np.clip(np.searchsorted(knots[j], x[:, j]), 0, 5)
            out += slopes[j][xs] * x[:, j]
        return out

    return f


def _teacher(d: int, depth: int, seed: int) -> Callable[[Array], Array]:
    rng = np.random.default_rng(seed)
    widths = [d] + [8] * max(1, depth) + [1]
    # non-negative weights ⇒ monotone; softplus activation ⇒ monotone
    ws = [rng.uniform(0.0, 1.0, size=(widths[i], widths[i + 1])) for i in range(len(widths) - 1)]
    bs = [rng.uniform(-0.5, 0.5, size=widths[i + 1]) for i in range(len(widths) - 1)]

    def f(x: Array) -> Array:
        h = x
        for i, (w, bnd) in enumerate(zip(ws, bs)):
            h = h @ w + bnd
            if i < len(ws) - 1:
                h = _softplus(h)
        return h[:, 0]

    return f


def _lattice(d: int, depth: int, seed: int) -> Callable[[Array], Array]:
    rng = np.random.default_rng(seed)
    # monotone linear "experts", then nested pairwise max/min of depth `depth`
    m = 2 ** max(1, depth)
    w = rng.uniform(0.0, 1.0, size=(d, m))
    bnd = rng.uniform(-0.5, 0.5, size=m)
    ops = rng.integers(0, 2, size=depth)  # 0=max, 1=min per level

    def f(x: Array) -> Array:
        h = x @ w + bnd  # (n, m), monotone in x
        for lvl in range(depth):
            half = h.shape[1] // 2
            a, b = h[:, :half], h[:, half : 2 * half]
            h = np.maximum(a, b) if ops[lvl] == 0 else np.minimum(a, b)
        return h[:, 0]

    return f


def _target_fn(kind: _Kind, *, c: int, d: int, seed: int) -> Callable[[Array], Array]:
    if kind == "additive":
        return _additive(d, seed)
    if kind == "teacher":
        return _teacher(d, c, seed)
    if kind == "lattice":
        return _lattice(d, c, seed)
    raise ValueError(f"unknown kind {kind!r}")


def _assert_monotone(f: Callable[[Array], Array], d: int, seed: int) -> None:
    rng = np.random.default_rng(seed + 999)
    x = rng.uniform(0, 1, size=(256, d))
    base = f(x)
    for j in range(d):
        xp = x.copy()
        xp[:, j] = np.minimum(1.0, x[:, j] + 0.05)
        if not (f(xp) - base >= -1e-9).all():
            raise AssertionError(f"target not monotone in dim {j}")


def synth_monotone(
    kind: _Kind,
    c: int,
    *,
    d: int = 6,
    n_train: int = 4000,
    n_test: int = 2000,
    seed: int = 0,
) -> DatasetBundle:
    """Build a monotone-regression :class:`DatasetBundle` (see module docstring).

    :param kind: Target family.
    :param c: Complexity knob (teacher/lattice depth; ignored for additive).
    :param d: Input dimension.
    :param n_train: Train rows; :param n_test: Test rows; :param seed: RNG seed.
    :returns: Regression bundle, all features monotone-increasing.
    """
    f = _target_fn(kind, c=c, d=d, seed=seed)
    _assert_monotone(f, d, seed)
    rng = np.random.default_rng(seed)
    x_tr = rng.uniform(0, 1, size=(n_train, d))
    x_te = rng.uniform(0, 1, size=(n_test, d))
    y_tr, y_te = f(x_tr), f(x_te)
    mu, sd = float(y_tr.mean()), float(y_tr.std() or 1.0)
    return DatasetBundle(
        name=f"synth-{kind}-{c}",
        task="regression",
        X_train=x_tr,
        y_train=(y_tr - mu) / sd,
        X_test=x_te,
        y_test=(y_te - mu) / sd,
        mono_increasing=tuple(range(d)),
        mono_decreasing=(),
        feature_names=tuple(f"x{j}" for j in range(d)),
        metadata={"kind": kind, "c": str(c)},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/benchmarks/test_synthetic.py -v && uv run mypy benchmarks/datasets/synthetic.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/datasets/synthetic.py tests/benchmarks/test_synthetic.py
git commit -m "bench: synthetic monotone-regression targets for the depth probe"
```

---

### Task 2: Probe runner (deep/shallow arms per (kind, c))

**Files:**
- Create: `benchmarks/monotone_depth_probe_run.py`
- Test: `tests/benchmarks/test_monotone_depth_probe_run.py`

**Interfaces:**
- Consumes: `synth_monotone` (Task 1); `search` + `final_eval` (`benchmarks._common.search`) — task-aware, return MSE for a regression bundle; `interquartile_mean` (`benchmarks._common.results`).
- Produces: `probe_dataset(kind, c, *, n_trials, search_seeds, final_seeds, epochs, backend="torch") -> dict` returning `{kind, c, deep_mse_iqm, shallow_mse_iqm, deep_values, shallow_values}` (raw per-seed MSE lists kept for the report's bootstrap). CLI: `--kinds`, `--cs`, budget flags, `--out`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/benchmarks/test_monotone_depth_probe_run.py
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks.monotone_depth_probe_run import probe_dataset


def test_probe_dataset_smoke() -> None:
    rec = probe_dataset("additive", c=1, n_trials=1, search_seeds=1, final_seeds=2, epochs=1)
    assert rec["kind"] == "additive" and rec["c"] == 1
    assert np.isfinite(rec["deep_mse_iqm"]) and np.isfinite(rec["shallow_mse_iqm"])
    assert len(rec["deep_values"]) == 2 and len(rec["shallow_values"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_monotone_depth_probe_run.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `monotone_depth_probe_run.py`**

```python
"""Depth probe: deep vs shallow monotone-residual on synthetic monotone targets.

For each (kind, c), run the standard search for the deep and shallow ``absolute``
residual arms on a synthetic monotone-regression bundle, refit + multi-seed test,
and record per-arm MSE IQM (+ raw per-seed values for the report's bootstrap).
See docs/superpowers/specs/2026-07-12-monotone-depth-synthetic-probe-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks._common.results import interquartile_mean
from benchmarks._common.search import final_eval, search
from benchmarks.datasets.synthetic import synth_monotone


def _arm_mse(bundle: Any, *, deep: bool, n_trials: int, search_seeds: int,
             final_seeds: int, epochs: int, backend: str) -> list[float]:
    study = search(bundle, mode="absolute", residual=True, deep=deep, backend=backend,
                   n_trials=n_trials, epochs=epochs, n_splits=1, search_seeds=search_seeds)
    agg = final_eval(bundle, study.best_params, mode="absolute", residual=True,
                     backend=backend, seeds=range(final_seeds), epochs=epochs)
    return [float(v) for v in agg.values]


def probe_dataset(kind: str, c: int, *, n_trials: int = 15, search_seeds: int = 2,
                  final_seeds: int = 8, epochs: int = 30, backend: str = "torch") -> dict[str, Any]:
    """Run both arms on ``synth_monotone(kind, c)``; return per-arm MSE IQMs + values."""
    bundle = synth_monotone(kind, c)  # type: ignore[arg-type]
    kw = dict(n_trials=n_trials, search_seeds=search_seeds, final_seeds=final_seeds,
              epochs=epochs, backend=backend)
    deep = _arm_mse(bundle, deep=True, **kw)
    shallow = _arm_mse(bundle, deep=False, **kw)
    return {
        "kind": kind, "c": c,
        "deep_mse_iqm": interquartile_mean(np.asarray(deep)),
        "shallow_mse_iqm": interquartile_mean(np.asarray(shallow)),
        "deep_values": deep, "shallow_values": shallow,
    }


def main() -> None:
    """CLI: sweep (kind, c) and write probe records JSON."""
    import argparse

    ap = argparse.ArgumentParser(description="monotone depth probe")
    ap.add_argument("--kinds", default="additive,teacher,lattice")
    ap.add_argument("--cs", default="1,2,4,8")
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--search-seeds", type=int, default=2)
    ap.add_argument("--final-seeds", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    recs = [
        probe_dataset(k, int(c), n_trials=args.n_trials, search_seeds=args.search_seeds,
                      final_seeds=args.final_seeds, epochs=args.epochs)
        for k in args.kinds.split(",")
        for c in args.cs.split(",")
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(recs, indent=2) + "\n")
    print(f"wrote {args.out} ({len(recs)} records)")  # noqa: T201


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run smoke test + mypy**

Run: `uv run pytest tests/benchmarks/test_monotone_depth_probe_run.py -v && uv run mypy benchmarks/monotone_depth_probe_run.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/monotone_depth_probe_run.py tests/benchmarks/test_monotone_depth_probe_run.py
git commit -m "bench: monotone depth-probe runner (deep/shallow arms on synthetic targets)"
```

---

### Task 3: Probe report (Δ(c) table + plot)

**Files:**
- Create: `benchmarks/_common/depth_probe_report.py`
- Test: `tests/benchmarks/test_depth_probe_report.py`

**Interfaces:**
- Consumes: probe records (Task 2 schema).
- Produces:
  - `delta_by_c(records, *, boot=2000, seed=0) -> list[dict]` — per record, `delta = shallow_mse_iqm − deep_mse_iqm` (positive ⇒ depth helps) with an independent-seed bootstrap band `delta_lo/delta_hi` over `deep_values`/`shallow_values` (mirror `size_ladder_report.delta_by_n`'s bootstrap, but on MSE with the flipped sign).
  - `probe_table(rows) -> str` (Markdown: kind, c, deep MSE, shallow MSE, Δ [CI]).
  - `render_probe_plot(rows, out_path) -> None` — Δ vs c, one line per `kind`, 0 reference line; writes `.png` + `.pdf`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_depth_probe_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks._common.depth_probe_report import delta_by_c, probe_table


def _rec(kind: str, c: int, deep: list[float], shallow: list[float]) -> dict[str, Any]:
    import numpy as np
    from benchmarks._common.results import interquartile_mean
    return {"kind": kind, "c": c,
            "deep_mse_iqm": interquartile_mean(np.asarray(deep)),
            "shallow_mse_iqm": interquartile_mean(np.asarray(shallow)),
            "deep_values": deep, "shallow_values": shallow}


def test_delta_by_c_sign_and_band() -> None:
    # deep clearly better (lower MSE) ⇒ positive delta
    rows = delta_by_c([_rec("teacher", 4, [0.10, 0.11, 0.10], [0.20, 0.21, 0.20])])
    r = rows[0]
    assert r["delta"] > 0
    assert r["delta_lo"] <= r["delta"] <= r["delta_hi"]


def test_probe_table_has_rows() -> None:
    rows = delta_by_c([_rec("additive", 1, [0.2, 0.2], [0.2, 0.2])])
    md = probe_table(rows)
    assert "| additive |" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_depth_probe_report.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `depth_probe_report.py`**

Model it on `benchmarks/_common/size_ladder_report.py` (read that file for the bootstrap + matplotlib idiom). `delta_by_c` resamples `deep_values` and `shallow_values` independently `boot` times (seeded `np.random.default_rng(seed)`), computes `iqm(shallow_boot) − iqm(deep_boot)` per draw, and takes the 2.5/97.5 percentiles for `delta_lo/hi`; `delta = shallow_mse_iqm − deep_mse_iqm`. `render_probe_plot` uses the Agg backend, one line per `kind` (Δ vs `c`), an axhline at 0, colorblind-safe colors, `bbox_inches="tight"`, and writes both `out_path` and `out_path.with_suffix(".pdf")` (clip yerr non-negative, as in `size_ladder_report`).

- [ ] **Step 4: Run tests + mypy**

Run: `uv run pytest tests/benchmarks/test_depth_probe_report.py -v && uv run mypy benchmarks/_common/depth_probe_report.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/depth_probe_report.py tests/benchmarks/test_depth_probe_report.py
git commit -m "bench: depth-probe report — Δ(c) table + plot (MSE, positive=depth helps)"
```

---

### Task 4: Docs page + RUNBOOK + toctree

**Files:**
- Create: `docs/benchmarks/monotone-depth-probe.md`, `benchmarks/RUNBOOK-depth-probe.md`
- Modify: `docs/benchmarks/index.md` (toctree)
- Test: `sphinx-build -W`

- [ ] **Step 1: Docs skeleton** — `monotone-depth-probe.md`: the theory framing (H-strong vs H-weak, the depth-separation-needs-oscillation argument, the DLN contrast) + a `{note}` that the table/plot are filled by the GPU run. **No image reference yet** (a missing image fails `-W`).
- [ ] **Step 2: RUNBOOK** — `RUNBOOK-depth-probe.md` mirroring `RUNBOOK-large-screen.md`: run `python -m benchmarks.monotone_depth_probe_run --kinds additive,teacher,lattice --cs 1,2,4,8 --out benchmarks/results/depth-probe/probe.json` (GPU-pin per `$MONONET_TORCH_DEVICE`; fixed deep/shallow bands first, iso-parameter frontier only where a signal appears); render via `depth_probe_report`; fill the docs page; commit results + plot.
- [ ] **Step 3: Wire toctree** — add `monotone-depth-probe` to `docs/benchmarks/index.md`.
- [ ] **Step 4: Verify** — `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html` → `build succeeded`, no warnings.
- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/monotone-depth-probe.md docs/benchmarks/index.md benchmarks/RUNBOOK-depth-probe.md
git commit -m "docs: monotone-depth-probe page skeleton + RUNBOOK + toctree"
```

---

## After the tasks (controller, per RUNBOOK)

GPU run: sweep `additive,teacher,lattice` × `c ∈ {1,2,4,8}` with the fixed deep/shallow bands (both GPUs, one `(kind,c)` process per slot, `n_jobs=1` — the screen's deadlock lesson); then the **iso-parameter frontier** only for families/`c` showing a signal. Render Δ(c) + fill the docs page; commit results + plot; PR.

## Self-review notes

- **Spec coverage:** families 1/2/3 → Task 1; deep/shallow arms + MSE → Task 2; Δ(c) with the shallow−deep sign + bootstrap → Task 3; docs/RUNBOOK + iso-param note → Task 4. Iso-parameter frontier + AUC-rescore of real data remain controller/follow-up per spec §6/§8.
- **Type consistency:** `probe_dataset` returns exactly the keys `delta_by_c`/`probe_table` consume (`deep_mse_iqm`, `shallow_mse_iqm`, `deep_values`, `shallow_values`, `kind`, `c`); `search`/`final_eval` are called with the same signature the loan/screen code uses; `synth_monotone` returns a `DatasetBundle` with `task="regression"` so `final_eval` scores MSE.
- **Deadlock guard:** the GPU run uses one process per `(kind,c)` at `n_jobs=1` (never threaded Optuna), per the screen finding.
