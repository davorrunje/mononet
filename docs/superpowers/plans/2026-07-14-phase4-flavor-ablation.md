# Phase 4: flavor-ablation benchmark (mixed · alternate · split) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`). **This plan is intended for a session on a GPU machine** — Tasks 1–4 (harness code) are CPU-testable and must land first; Tasks 5–6 are the GPU run + write-up.

**Goal:** Measure, on real data + a synthetic depth ladder, whether `alternate` + composition-aware init trains deep *plain* monotone stacks where `mixed`/`split` diverge (H-plain), whether composition-aware init beats the legacy per-layer init per activation (H-init), and confirm residual is a wash (H-residual).

**Architecture:** A **standalone fixed-architecture grid sweep** (`benchmarks/flavor_ablation.py`) that enumerates cells `(dataset, flavor, activation, depth[, alt_init])`, builds a `BenchmarkConfig` per cell at **fixed** width/LR/seeds, calls the existing `benchmarks._common.runner.run(cfg, bundle)` (which loops seeds), and aggregates primary metric + dispersion + convergence + **divergence-rate**. It reuses `build_model`, `runner.run`, `_score_predictions`, the dataset registry, and the dual-GPU launcher pattern. It does **not** touch the Optuna flavor-search path (`_ALL_FLAVORS`/`flavor_name`/`_parse_flavors`/`suggest_config`). See spec [`2026-07-14-flavor-ablation-benchmark-design.md`](../specs/2026-07-14-flavor-ablation-benchmark-design.md).

**Tech Stack:** Python 3.11+, uv, pytest; PyTorch / JAX (Flax NNX) / Keras 3; Optuna NOT used here; matplotlib for the docs plots.

## Global Constraints

- **Hard dependency:** requires **PR #109 (phase 2: `mode="alternate"` + `prev=`)** merged to `main`. Before starting, `git checkout main && git pull` and confirm `from mononet.torch import MonoLinear; MonoLinear(4, 8, mode="alternate", activation="relu")` works. If #109 is not yet merged, stop and report.
- Branch `feat/phase4-flavor-ablation` (rebase on `main` once #109 lands). Commit `git commit --no-gpg-sign`. Never commit to `main`.
- Python 3.11+, line length 88 (ruff). Strict mypy (`uv run mypy --group bench` covers `benchmarks/`). MyST field-list docstrings on public functions.
- The published wheel ships layers only — all benchmark code lives under `benchmarks/`, never in `mononet/`.
- **No `mononet/` change** in this phase (the `alternate` layer already exists from #109). Phase 4 is `benchmarks/` + `docs/` only.
- GPU steps (5–6) run from a `gpu-torch` (or `gpu-jax`) devcontainer; the bench group is installed via `uv sync --group bench --extra all-cpu` (or the GPU extra). CPU steps (1–4) run in the default devcontainer.

## Pinned decisions (from spec §5, §9)

- **Grid cell** = `(dataset, mode, alt_init, activation, depth)`, topology **plain** for the focused run. `mode ∈ {mixed, split, alternate}`; `alt_init ∈ {None, "composition", "legacy"}` (non-None only for `alternate`). Fixed per cell: `width=32`, base `lr=1e-3`, `seeds=range(5)`, `epochs=300` with early stopping, `embed_hidden=(32,)`.
- **`alt_init="composition"`** → build with `mode="alternate"` + `prev=` chaining (real construction). **`alt_init="legacy"`** → build the same pure-class layers as `mode="mixed"` with `convex_fraction` alternating `1.0,0.0,1.0,…` per depth index (the collapse baseline; no new layer code).
- **Divergence:** a per-seed run is `diverged` if final val loss is non-finite or `> 10 ×` the predict-the-mean baseline. `divergence_rate` per cell = fraction of seeds diverged.
- **Focused-first datasets:** `heart`, `auto`, `synth_lattice_clow`, `synth_lattice_cmid`, `synth_lattice_chigh`. Activations: `relu, elu, softplus, selu`. Depths: `4, 8, 16`. All flavor cells.
- **LR mini-sweep:** the same grid runner with `lr ∈ {1e-4,3e-4,1e-3,3e-3,1e-2}`, on `heart` + `auto` only, at one depth (`8`).
- **Head:** the monotone read-out head stays `mode="mixed"` (linear identity) even when the stack is `alternate` (the head is parity-neutral and takes no `prev`).
- **Residual-alternate** is **out of scope for the focused run** (plain only); Task 1 guards `cfg.mode=="alternate" and cfg.residual` with `NotImplementedError`. It is a documented expansion (spec §4) implemented only if the focused pass warrants residual.

---

### Task 1: `BenchmarkConfig` + `model_builder` support for `alternate`

**Files:**
- Modify: `benchmarks/_common/config.py` (`mode` Literal line 66; add `alt_init` field ~line 71; docstring)
- Modify: `benchmarks/_common/model_builder.py` (`_build_torch_stack` 40-89, `_build_torch` head 137; `_build_jax_stack` 191-243, head 277; keras stack 365-386, head 390)
- Test: `tests/benchmarks/test_model_builder_alternate.py` (new)

**Interfaces:**
- Produces: `BenchmarkConfig(..., mode: Literal["split","mixed","alternate"], alt_init: Literal["composition","legacy"] | None = None)`; `build_model(cfg, bundle)` builds a plain alternate stack for `cfg.mode=="alternate"` (composition via `prev=`, or legacy via `mode="mixed"` alternating `cf`).

- [ ] **Step 1: Write failing builder tests**

Create `tests/benchmarks/test_model_builder_alternate.py` (torch shown; add jax + keras analogues):
```python
import numpy as np
import pytest

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.model_builder import build_model
from benchmarks._common.bundle import DatasetBundle


def _bundle():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (64, 3)).astype("float32")
    y = x.sum(1, keepdims=True).astype("float32")
    return DatasetBundle(
        name="t", task="regression", feature_names=["a", "b", "c"],
        mono=[1, 1, 1], x_train=x, y_train=y, x_val=x, y_val=y, x_test=x, y_test=y,
    )  # adapt to the real DatasetBundle constructor (see benchmarks/_common/bundle.py)


def _cfg(mode, alt_init, depth=4, backend="torch"):
    return BenchmarkConfig(
        dataset="t", backend=backend, mode=mode, residual=False, depth=depth,
        width=16, activation="relu", convex_fraction=0.5, embed_hidden=(),
        dropout=0.0, optimizer=OptimizerSpec(name="adam", lr=1e-3),
        lr_decay=None, batch_size=64, epochs=1, early_stopping=None,
        seeds=(0,), metrics=("mse",), alt_init=alt_init,
    )


def test_alternate_composition_builds_and_is_finite() -> None:
    import torch
    m = build_model(_cfg("alternate", "composition"), _bundle())
    out = m(torch.zeros(2, 3))
    assert torch.isfinite(out).all()


def test_alternate_legacy_builds_pure_mixed_layers() -> None:
    import torch
    m = build_model(_cfg("alternate", "legacy"), _bundle())
    # legacy arm uses mode="mixed" pure-class layers (convex_fraction 1/0 alternating)
    monolinears = [mod for mod in m.modules() if getattr(mod, "mode", None) == "mixed"
                   and getattr(mod, "convex_fraction", None) in (0.0, 1.0)]
    assert len(monolinears) >= 4  # the 4 alternating layers
    assert torch.isfinite(m(torch.zeros(2, 3))).all()


def test_residual_alternate_not_supported() -> None:
    cfg = _cfg("alternate", "composition")
    cfg = cfg.replace(residual=True)
    with pytest.raises(NotImplementedError, match="residual"):
        build_model(cfg, _bundle())
```
(Read `benchmarks/_common/bundle.py` for the exact `DatasetBundle` constructor and adjust `_bundle`.)

- [ ] **Step 2: Run to verify failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_model_builder_alternate.py -v`
Expected: FAIL (`BenchmarkConfig` has no `alt_init`; builder has no alternate branch).

- [ ] **Step 3: Config**

`benchmarks/_common/config.py`: change line 66 to `mode: Literal["split", "mixed", "alternate"]`; add after `convex_fraction` (line 71) a field `alt_init: Literal["composition", "legacy"] | None = None`. Because the dataclass is `frozen, slots` and other fields lack defaults, put `alt_init` **last** (after `metrics`) to keep the no-default ordering valid, or give it a default and move it after all non-default fields. Update the docstring. Update `benchmarks/_common/config_io.py:load_config` to accept/pass `alt_init` (default `None`) and widen its `mode` Literal (line 19).

- [ ] **Step 4: torch builder alternate branch**

In `_build_torch_stack` (`model_builder.py:40-89`), add before the `if cfg.residual:` (so alternate is handled first):
```python
    if cfg.mode == "alternate":
        if cfg.residual:
            raise NotImplementedError("residual + alternate not supported (plain only)")
        prev_layer = None
        for i in range(cfg.depth):
            if cfg.alt_init == "legacy":
                cf = 1.0 if i % 2 == 0 else 0.0
                lay = MonoLinear(prev, cfg.width, mode="mixed",
                                 activation=cfg.activation, convex_fraction=cf)
            else:  # composition
                lay = MonoLinear(prev, cfg.width, mode="alternate",
                                 activation=cfg.activation, prev=prev_layer)
                prev_layer = lay
            mono_layers.append(lay)
            prev = cfg.width
        return nn.Sequential(*mono_layers), prev
```
In `_build_torch` (line 137) make the head non-alternate:
```python
    head_mode = "mixed" if cfg.mode == "alternate" else cfg.mode
    self.head = MonoLinear(stack_out, 1, mode=head_mode, activation="identity")
```

- [ ] **Step 5: jax + keras builder branches**

Mirror in `_build_jax_stack` (`:191-243`, thread `rngs=rngs` on each `MonoLinear`; `prev_layer` chaining) and the inline keras stack (`:365-386`, functional `z = MonoDense(...)(z)` — for composition, keras `MonoDense` takes `prev=` the previous `MonoDense` **instance**, so keep a reference: `layer = MonoDense(cfg.width, mode="alternate", activation=cfg.activation, prev=prev_layer); z = layer(z); prev_layer = layer`). Set the head mode to `"mixed"` when `cfg.mode=="alternate"` in `_build_jax` (`:277`) and `_build_keras` (`:390`). Add the `NotImplementedError` residual guard in each.

- [ ] **Step 6: Run builder tests (all backends) green**

```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_model_builder_alternate.py -v
MONONET_TEST_BACKEND=jax   uv run pytest tests/benchmarks/test_model_builder_alternate.py -v
MONONET_TEST_BACKEND=keras uv run pytest tests/benchmarks/test_model_builder_alternate.py -v
```
Expected: PASS. Also run the existing `tests/benchmarks/test_model_builder_*` to confirm no regression.

- [ ] **Step 7: Commit**

```bash
git add benchmarks/_common/config.py benchmarks/_common/config_io.py benchmarks/_common/model_builder.py tests/benchmarks/test_model_builder_alternate.py
git commit --no-gpg-sign -m "feat(bench): alternate construction in the model builder (composition + legacy)"
```

---

### Task 2: Divergence detection in the runner

**Files:**
- Modify: `benchmarks/_common/results.py` (`ResultRow` 15-35: add `diverged: bool`)
- Modify: `benchmarks/_common/runner.py` (`run` 41-73: compute + pass `diverged`; the 3 `_train_*` loops can surface a final-loss signal, or compute post-hoc in `run` from the val prediction)
- Test: `tests/benchmarks/test_divergence.py` (new)

**Interfaces:**
- Produces: `ResultRow(..., diverged: bool)`; a run is `diverged` when final val loss is non-finite or `> 10 × baseline` (`baseline = Var[y_val]` for regression; majority-class log-loss for classification).

- [ ] **Step 1: Write failing test**

`tests/benchmarks/test_divergence.py`: construct a `ResultRow` with `diverged=True/False` and assert the field exists + round-trips in whatever serialization `results.py` uses; and a unit test of the divergence predicate helper (below) — `is_diverged(final_loss=float("nan"), baseline=1.0) is True`, `is_diverged(0.5, 1.0) is False`, `is_diverged(20.0, 1.0) is True`.

- [ ] **Step 2: Run to verify failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_divergence.py -v` → FAIL.

- [ ] **Step 3: Implement**

Add `diverged: bool = False` to `ResultRow` (`results.py:15-35`). In `runner.py`, add a helper:
```python
def is_diverged(final_loss: float, baseline: float) -> bool:
    """A run diverged if its final loss is non-finite or exceeds 10x the predict-the-mean baseline."""
    import math
    return (not math.isfinite(final_loss)) or (final_loss > 10.0 * baseline)
```
In `run` (`:41-73`), after `_evaluate`, compute the val-loss baseline for the bundle (regression: `float(np.var(y_val))`; classification: majority-class binary cross-entropy) and the run's final val loss (reuse the eval `mse`/log-loss), then pass `diverged=is_diverged(final_val_loss, baseline)` into the `ResultRow(...)` at `:62-72`. Keep the existing `scores`/`epochs_run` behavior.

- [ ] **Step 4: Run green + commit**

```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_divergence.py tests/benchmarks/test_runner.py -v
git add benchmarks/_common/results.py benchmarks/_common/runner.py tests/benchmarks/test_divergence.py
git commit --no-gpg-sign -m "feat(bench): per-run divergence flag on ResultRow"
```

---

### Task 3: The grid runner `benchmarks/flavor_ablation.py`

**Files:**
- Create: `benchmarks/flavor_ablation.py`
- Test: `tests/benchmarks/test_flavor_ablation.py` (new)

**Interfaces:**
- Consumes: `build_model`, `runner.run` (Tasks 1–2), the dataset registry (`benchmarks.datasets.registry.load`), `benchmarks._common.results`.
- Produces: `ablation_cells(focused: bool) -> list[Cell]`; `run_cell(cell, backend, lr, bundle) -> dict` (aggregated record); `run_dataset_ablation(dataset, backend, *, lr_sweep, out_dir, smoke) -> Path`; a Typer/argparse CLI `python -m benchmarks.flavor_ablation --dataset heart --backend torch --out-dir benchmarks/results/flavor-ablation`.

- [ ] **Step 1: Write the failing smoke test**

`tests/benchmarks/test_flavor_ablation.py`:
```python
from pathlib import Path

from benchmarks.flavor_ablation import ablation_cells, run_dataset_ablation


def test_focused_cells_cover_all_flavors() -> None:
    cells = ablation_cells(focused=True)
    flavors = {(c.mode, c.alt_init) for c in cells}
    assert flavors == {("mixed", None), ("split", None),
                       ("alternate", "composition"), ("alternate", "legacy")}
    assert {c.activation for c in cells} == {"relu", "elu", "softplus", "selu"}
    assert {c.depth for c in cells} == {4, 8, 16}


def test_smoke_run_writes_records_with_divergence(tmp_path: Path) -> None:
    # smoke: a tiny synthetic dataset, 1 seed, 2 epochs — asserts schema, not science.
    out = run_dataset_ablation("synth_lattice_clow", "torch",
                               lr_sweep=False, out_dir=tmp_path, smoke=True)
    import json
    recs = json.loads(out.read_text())
    assert recs, "no records written"
    r = recs[0]
    for key in ("dataset", "mode", "alt_init", "activation", "depth",
                "metric_iqm", "metric_iqr", "epochs_median",
                "divergence_rate", "collapsed", "n_seeds"):
        assert key in r, f"missing {key}"
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement the grid runner**

`benchmarks/flavor_ablation.py`:
```python
"""Fixed-architecture flavor ablation grid (mixed / alternate / split).

A standalone grid sweep (NOT the Optuna flavor-search path). Each cell trains at a
fixed architecture over N seeds via benchmarks._common.runner.run, and aggregates
primary metric + dispersion + convergence + divergence-rate. See
docs/superpowers/specs/2026-07-14-flavor-ablation-benchmark-design.md.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from benchmarks._common.config import BenchmarkConfig, OptimizerSpec, EarlyStoppingSpec
from benchmarks._common.runner import run
from benchmarks.datasets.registry import load

_ACTS = ("relu", "elu", "softplus", "selu")
_DEPTHS = (4, 8, 16)
_FLAVORS: tuple[tuple[str, str | None], ...] = (
    ("mixed", None), ("split", None),
    ("alternate", "composition"), ("alternate", "legacy"),
)
_FOCUSED_DATASETS = ("heart", "auto",
                     "synth_lattice_clow", "synth_lattice_cmid", "synth_lattice_chigh")
_LR_GRID = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass(frozen=True, slots=True)
class Cell:
    mode: str
    alt_init: str | None
    activation: str
    depth: int


def ablation_cells(focused: bool = True) -> list[Cell]:
    """Enumerate the plain-topology grid cells (focused = the first GPU pass)."""
    return [
        Cell(mode, alt_init, act, depth)
        for (mode, alt_init) in _FLAVORS
        for act in _ACTS
        for depth in _DEPTHS
    ]


def _cell_config(dataset: str, backend: str, cell: Cell, lr: float,
                 *, smoke: bool) -> BenchmarkConfig:
    seeds = (0,) if smoke else tuple(range(5))
    epochs = 2 if smoke else 300
    return BenchmarkConfig(
        dataset=dataset, backend=backend, mode=cell.mode, residual=False,
        depth=cell.depth, width=32, activation=cell.activation, convex_fraction=0.5,
        embed_hidden=(32,), dropout=0.0,
        optimizer=OptimizerSpec(name="adam", lr=lr), lr_decay=None,
        batch_size=256, epochs=epochs,
        early_stopping=None if smoke else EarlyStoppingSpec(monitor="val", patience=30),
        seeds=seeds,
        metrics=("roc_auc", "accuracy") if _is_classification(dataset) else ("mse", "rmse"),
        alt_init=cell.alt_init,
    )


def _is_classification(dataset: str) -> bool:
    return load(dataset).task == "binary_classification"


def _aggregate(rows: list, primary: str) -> dict:
    vals = np.array([r.scores[primary] for r in rows], dtype=float)
    finite = vals[np.isfinite(vals)]
    iqm = float(np.mean(np.sort(finite)[len(finite)//4: len(finite)-len(finite)//4])) \
        if len(finite) >= 4 else float(np.mean(finite)) if len(finite) else float("nan")
    epochs = np.array([r.epochs_run for r in rows], dtype=float)
    return {
        "metric_iqm": iqm,
        "metric_iqr": float(np.subtract(*np.percentile(finite, [75, 25]))) if len(finite) else float("nan"),
        "epochs_median": float(np.median(epochs)),
        "divergence_rate": float(np.mean([r.diverged for r in rows])),
        "n_seeds": len(rows),
    }


def run_cell(dataset: str, backend: str, cell: Cell, lr: float, bundle,
             *, smoke: bool) -> dict:
    cfg = _cell_config(dataset, backend, cell, lr, smoke=smoke)
    primary = "roc_auc" if bundle.task == "binary_classification" else "mse"
    # collapse pre-check at init (before training): build once, check finite + non-zero var.
    collapsed = _collapse_precheck(cfg, bundle, backend)
    rows = run(cfg, bundle)
    rec = {"dataset": dataset, "mode": cell.mode, "alt_init": cell.alt_init,
           "activation": cell.activation, "depth": cell.depth, "lr": lr,
           "primary": primary, "collapsed": collapsed}
    rec.update(_aggregate(rows, primary))
    return rec


def run_dataset_ablation(dataset: str, backend: str, *, lr_sweep: bool,
                         out_dir: Path, smoke: bool = False) -> Path:
    bundle = load(dataset)
    cells = ablation_cells(focused=True)
    lrs = _LR_GRID if lr_sweep else (1e-3,)
    if lr_sweep:  # mini-sweep: fix depth=8, sweep lr
        cells = [c for c in cells if c.depth == 8]
    recs = [run_cell(dataset, backend, c, lr, bundle, smoke=smoke)
            for c in cells for lr in lrs]
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-lrsweep" if lr_sweep else ""
    path = out_dir / f"{dataset}{suffix}.json"
    path.write_text(json.dumps(recs, indent=2) + "\n")
    return path
```
Implement `_collapse_precheck(cfg, bundle, backend)`: build the model via `build_model`, run a forward on a small batch, return `True` if output variance ≈ 0 or all-constant (mirror the init-time check from `mononet`'s phase-2 tests). Add a `main()` CLI (argparse or Typer, matching `benchmarks/search.py` style) exposing `--dataset --backend --lr-sweep --out-dir --smoke`.

- [ ] **Step 4: Run the smoke test green**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_flavor_ablation.py -v`
Expected: PASS (the `synth_lattice_clow` smoke completes fast at 1 seed / 2 epochs).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/flavor_ablation.py tests/benchmarks/test_flavor_ablation.py
git commit --no-gpg-sign -m "feat(bench): flavor-ablation grid runner (metric/dispersion/convergence/divergence)"
```

---

### Task 4: Dual-GPU launcher + full-repo green

**Files:**
- Create: `benchmarks/flavor_ablation_launch.py` (adapt `benchmarks/stage2_launch.py`'s device-pool pattern)
- Test: `tests/benchmarks/test_flavor_ablation_launch.py` (round-robin device assignment, `--dry-run`)

- [ ] **Step 1** Adapt `stage2_launch.py` (`_run_dataset` spawns `python -m benchmarks.flavor_ablation --dataset <name> --backend <b> --out-dir benchmarks/results/flavor-ablation` with `MONONET_TORCH_DEVICE=<device>`; `run_parallel` round-robins `--devices cuda:0 cuda:1`). Add a `--dry-run` that prints the command matrix without spawning. TDD the round-robin + dry-run like `tests/benchmarks/test_*launch*` if one exists.

- [ ] **Step 2** Full-repo green (CPU):
```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks -v
uv run ruff check --exit-non-zero-on-fix && uv run ruff format --check && uv run mypy
uv run pre-commit run --all-files
```
Expected: PASS.

- [ ] **Step 3** Commit, push, open the **draft PR** (see "PR" below).

---

### Task 5 (GPU): focused ablation run

- [ ] **Step 1** On a `gpu-torch` devcontainer, from `main` with this branch rebased and #109 merged: `uv sync --group bench`.
- [ ] **Step 2** Run the focused sweep across the 5 datasets, dual-GPU:
```bash
python -m benchmarks.flavor_ablation_launch --datasets heart auto \
  synth_lattice_clow synth_lattice_cmid synth_lattice_chigh \
  --backend torch --devices cuda:0 cuda:1 --out-dir benchmarks/results/flavor-ablation
```
- [ ] **Step 3** LR mini-sweep on the 2 real datasets:
```bash
python -m benchmarks.flavor_ablation --dataset heart --backend torch --lr-sweep --out-dir benchmarks/results/flavor-ablation
python -m benchmarks.flavor_ablation --dataset auto  --backend torch --lr-sweep --out-dir benchmarks/results/flavor-ablation
```
- [ ] **Step 4** Commit the results JSONs: `git add benchmarks/results/flavor-ablation && git commit --no-gpg-sign -m "results(bench): focused flavor-ablation run"`.

---

### Task 6 (GPU/CPU): write the three benchmark docs

- [ ] **Step 1** From the committed results, render three pages under `docs/benchmarks/` (a small `make_tables`-style script or notebook): (1) **flavor study** (mixed vs alternate-composition vs split × activation × depth: primary metric + IQR), (2) **initialization study** (divergence-rate vs depth per flavor; alternate composition-vs-legacy per activation — the collapse made visible), (3) **residual study** (only if the residual expansion was run; otherwise a one-paragraph "focused run was plain-only" note). Read the H-plain / H-init / H-residual verdicts off the numbers.
- [ ] **Step 2** Docs build `-W` green; commit; mark the PR ready for review.
- [ ] **Step 3 (expansion, optional)** If the focused pass is informative, extend `_FOCUSED_DATASETS` to the full Stage-2 set and implement the residual-alternate arm (custom `F=[convex, concave]` per block, `prev=`-chained, near-zero last layer — spec §5.2), then re-run.

---

## PR (draft, for the GPU session)

Open as **draft** after Task 4 with a body that gives the GPU session everything: the goal + the three hypotheses (H-plain/H-init/H-residual), the pinned grid (datasets/flavors/activations/depths/LR), the `alt_init` composition-vs-legacy meaning, the exact run commands (Tasks 5–6), the `#109` dependency, and links to this plan + the spec. Title: `feat(bench): flavor-ablation benchmark (Phase 4) — harness + GPU run`.

## Notes

- Phase 3 (concepts-docs distillation) and the notebook release-gate remain separate.
- If `runner.run`'s early-stopping monitor key differs from `"val"`, use the repo's actual monitor token (check `EarlyStoppingSpec` usage in `runner.py`).
