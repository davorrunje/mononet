# HP-search Sensitivity Curves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a committed benchmark diagnostic that shows, per `(dataset, flavor)`, how the tuned result evolves with Optuna trial count — the search-objective best-so-far curve (free from committed storage) and the test-of-incumbent curve (bounded re-eval) — plus a saturation table.

**Architecture:** A pure analysis+render module (`benchmarks/_common/sensitivity_report.py`) reads committed Optuna storage DBs, computes the best-so-far trajectory and a saturation point, reconstructs the test-of-incumbent step function by re-evaluating only the best-so-far changepoints via the existing `search.final_eval`, and renders a faceted PNG+PDF figure mirroring `size_ladder_report.render_plot`. A thin CLI (`benchmarks/sensitivity.py`) drives it and prints a Markdown saturation table. A prospective `log_test_trajectory` flag makes the test curve free for future runs. A docs page presents the figure + table.

**Tech Stack:** Python 3.11+, Optuna (storage read-back), NumPy, matplotlib (Agg), pytest. Benchmark-only code under `benchmarks/` (not shipped in the wheel).

## Global Constraints

- **Prerequisite:** PR #111 must be merged first — it provides the committed Optuna storage DBs at `benchmarks/results/alternate-base/studies/{dataset}-{flavor}.db` and the `final_eval(..., embed_layers=…)` signature this plan calls. Branch this work off `main` after #111 merges (or rebase onto it).
- **No re-training of the search.** Curve A is read from storage. Curve B re-evaluates only the *distinct best-so-far incumbents* (a handful per study), never all trials — and the count of re-evaluated configs is logged, never silently hidden.
- The base-result studies were run with `embed_layers=2`; every `final_eval` re-eval in this plan passes `embed_layers=2`.
- Figures are committed as **both** `.png` (docs) and `.pdf` (paper), from one Agg render, no title — mirror `benchmarks/_common/size_ladder_report.py:render_plot`.
- MyST field-list docstrings (`:param:`/`:returns:`) on all public functions; types from signature annotations only. Strict mypy; line length 88 (ruff). Stdlib dataclasses only — no Pydantic.
- Metric direction comes from `benchmarks._common.search._lower_is_better(metric)`; `blog` values are stored as MSE and reported as RMSE (`sqrt`) — match `make_tables`/`size_ladder_report`.

---

### Task 1: Best-so-far trajectory + saturation point

**Files:**
- Create: `benchmarks/_common/sensitivity_report.py`
- Test: `tests/benchmarks/test_sensitivity_report.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `best_so_far(values: list[float], lower: bool) -> list[float]`; `saturation_trial(traj: list[float], lower: bool, p: float = 0.99) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_sensitivity_report.py
import pytest
pytest.importorskip("optuna")
from benchmarks._common.sensitivity_report import best_so_far, saturation_trial


def test_best_so_far_minimize_is_cumulative_min():
    assert best_so_far([5.0, 3.0, 4.0, 1.0, 2.0], lower=True) == [5.0, 3.0, 3.0, 1.0, 1.0]


def test_best_so_far_maximize_is_cumulative_max():
    assert best_so_far([0.1, 0.3, 0.2, 0.9], lower=False) == [0.1, 0.3, 0.3, 0.9]


def test_saturation_trial_reaches_fraction_of_gain():
    # gain 5->1 = 4; 99% gain = reach <= 1 + 0.01*4 = 1.04; first at index 3 (1-based 4)
    traj = [5.0, 3.0, 3.0, 1.0, 1.0]
    assert saturation_trial(traj, lower=True, p=0.99) == 4


def test_saturation_trial_flat_trajectory_is_one():
    assert saturation_trial([2.0, 2.0, 2.0], lower=True) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py -q`
Expected: FAIL — `ModuleNotFoundError: benchmarks._common.sensitivity_report`.

- [ ] **Step 3: Write the implementation**

```python
# benchmarks/_common/sensitivity_report.py
"""Reconstruct HP-search sensitivity curves from committed Optuna storage.

Curve A (best-so-far search objective) is read directly from storage. Curve B
(test metric of the running incumbent) re-evaluates only the best-so-far
changepoints via `benchmarks._common.search.final_eval` — a bounded re-eval,
never a re-run of the search. See
`docs/superpowers/specs/2026-07-15-hp-search-sensitivity-curves-design.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def best_so_far(values: list[float], lower: bool) -> list[float]:
    """Cumulative best of a per-trial objective sequence.

    :param values: Per-trial objective values, in trial order.
    :param lower: Whether a lower objective is better (running `min`, else `max`).
    :returns: The running-best sequence, same length as ``values``.
    """
    out: list[float] = []
    best = None
    for v in values:
        best = v if best is None else (min(best, v) if lower else max(best, v))
        out.append(best)
    return out


def saturation_trial(traj: list[float], lower: bool, p: float = 0.99) -> int:
    """Smallest 1-based trial count reaching fraction ``p`` of the eventual gain.

    With ``G = |traj[-1] - traj[0]|`` the total improvement, returns the first
    ``t`` (1-based) where ``|traj[-1] - traj[t-1]| <= (1 - p) * G``. A flat
    trajectory (``G == 0``) returns ``1``.

    :param traj: A best-so-far trajectory (monotone), as from :func:`best_so_far`.
    :param lower: Whether lower is better (unused for the gap magnitude; kept for
        a symmetric call site and future signed variants).
    :param p: Fraction of the eventual gain to reach (default 0.99).
    :returns: The 1-based saturation trial count ``t*``.
    """
    if not traj:
        return 0
    final = traj[-1]
    gain = abs(final - traj[0])
    if gain == 0.0:
        return 1
    tol = (1.0 - p) * gain
    for i, b in enumerate(traj):
        if abs(final - b) <= tol:
            return i + 1
    return len(traj)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/sensitivity_report.py tests/benchmarks/test_sensitivity_report.py
git commit --no-gpg-sign -m "bench(sensitivity): best-so-far trajectory + saturation point"
```

---

### Task 2: Load studies and extract incumbent changepoints

**Files:**
- Modify: `benchmarks/_common/sensitivity_report.py`
- Test: `tests/benchmarks/test_sensitivity_report.py`

**Interfaces:**
- Consumes: `best_so_far` (Task 1).
- Produces: `load_study(db_path: Path, study_name: str)`; `completed_values(study, lower) -> list[float]`; `incumbent_changepoints(study, lower) -> list[tuple[int, dict]]` returning `(1-based trial index, params)` at each best-so-far improvement.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/benchmarks/test_sensitivity_report.py
from benchmarks._common.sensitivity_report import (
    completed_values,
    incumbent_changepoints,
)


class _FakeTrial:
    def __init__(self, value, params, complete=True):
        import optuna
        self.value = value
        self.params = params
        self.state = (
            optuna.trial.TrialState.COMPLETE if complete
            else optuna.trial.TrialState.FAIL
        )
        self.user_attrs: dict = {}


class _FakeStudy:
    def __init__(self, trials):
        self.trials = trials


def test_completed_values_skips_non_complete():
    s = _FakeStudy([_FakeTrial(1.0, {}), _FakeTrial(None, {}, complete=False)])
    assert completed_values(s, lower=True) == [1.0]


def test_incumbent_changepoints_are_the_improving_trials():
    s = _FakeStudy([
        _FakeTrial(5.0, {"depth": 1}),
        _FakeTrial(3.0, {"depth": 2}),
        _FakeTrial(4.0, {"depth": 3}),
        _FakeTrial(1.0, {"depth": 4}),
    ])
    cps = incumbent_changepoints(s, lower=True)
    assert [i for i, _ in cps] == [1, 2, 4]
    assert cps[-1][1] == {"depth": 4}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'completed_values'`.

- [ ] **Step 3: Write the implementation**

```python
# add to benchmarks/_common/sensitivity_report.py

import optuna


def load_study(db_path: "Path", study_name: str) -> optuna.Study:
    """Load an Optuna study from a committed sqlite storage file (read-only use).

    :param db_path: Path to the ``{dataset}-{flavor}.db`` sqlite file.
    :param study_name: The study name it was created under (``{dataset}-{flavor}``).
    :returns: The loaded :class:`optuna.Study`.
    """
    return optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")


def completed_values(study: optuna.Study, lower: bool) -> list[float]:
    """Objective values of COMPLETE trials, in trial order.

    :param study: A loaded study (or a duck-typed stand-in with ``.trials``).
    :param lower: Whether lower is better (accepted for call-site symmetry).
    :returns: The per-trial objective values for completed trials only.
    """
    return [
        float(t.value)
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]


def incumbent_changepoints(
    study: optuna.Study, lower: bool
) -> list[tuple[int, dict]]:
    """`(1-based trial index, params)` at each best-so-far improvement.

    Iterates completed trials in order; emits a changepoint whenever the running
    best strictly improves (including the first completed trial).

    :param study: A loaded study (or duck-typed stand-in with ``.trials``).
    :param lower: Whether a lower objective is better.
    :returns: The improving trials as ``(index, params)`` pairs, in order.
    """
    out: list[tuple[int, dict]] = []
    best: float | None = None
    idx = 0
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None:
            continue
        idx += 1
        v = float(t.value)
        improved = (
            best is None or (v < best if lower else v > best)
        )
        if improved:
            best = v
            out.append((idx, dict(t.params)))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/sensitivity_report.py tests/benchmarks/test_sensitivity_report.py
git commit --no-gpg-sign -m "bench(sensitivity): load study + incumbent changepoints"
```

---

### Task 3: Test-of-incumbent curve via bounded re-eval

**Files:**
- Modify: `benchmarks/_common/sensitivity_report.py`
- Test: `tests/benchmarks/test_sensitivity_report.py`

**Interfaces:**
- Consumes: `incumbent_changepoints` (Task 2); `benchmarks._common.search.final_eval`, `benchmarks._common.bundle.DatasetBundle`.
- Produces: `incumbent_test_curve(study, bundle, *, mode, residual, backend, lower, n_trials, seeds, embed_layers=2) -> tuple[list[float], int]` returning the per-trial test metric of the running incumbent (length `n_trials`, step-held between changepoints) and the count of distinct incumbents re-evaluated. Prefers a stored `test_metric` user-attr when present, else re-evaluates.

- [ ] **Step 1: Write the failing test** (monkeypatch `final_eval` so no training runs)

```python
# append to tests/benchmarks/test_sensitivity_report.py
from types import SimpleNamespace
import benchmarks._common.sensitivity_report as sr


def test_incumbent_test_curve_reevaluates_once_per_incumbent(monkeypatch):
    study = _FakeStudy([
        _FakeTrial(5.0, {"depth": 1}),
        _FakeTrial(3.0, {"depth": 2}),
        _FakeTrial(4.0, {"depth": 3}),
    ])
    calls = {"n": 0}

    def fake_final_eval(bundle, params, **kw):
        calls["n"] += 1
        # test metric keyed off a param so we can assert the step-hold
        agg = SimpleNamespace(metric=float(params["depth"]))
        return agg, []

    monkeypatch.setattr(sr, "final_eval", fake_final_eval)
    curve, n_eval = sr.incumbent_test_curve(
        study, bundle=object(), mode="split", residual=False, backend="torch",
        lower=True, n_trials=3, seeds=range(1),
    )
    # incumbents at trials 1 (depth1) and 2 (depth2); trial 3 holds depth2
    assert curve == [1.0, 2.0, 2.0]
    assert n_eval == 2
    assert calls["n"] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py::test_incumbent_test_curve_reevaluates_once_per_incumbent -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'incumbent_test_curve'`.

- [ ] **Step 3: Write the implementation**

```python
# add to benchmarks/_common/sensitivity_report.py
from typing import Any, Iterable

from benchmarks._common.search import final_eval


def incumbent_test_curve(
    study: optuna.Study,
    bundle: Any,
    *,
    mode: str,
    residual: bool,
    backend: str,
    lower: bool,
    n_trials: int,
    seeds: Iterable[int],
    embed_layers: int = 2,
) -> tuple[list[float], int]:
    """Test metric of the running incumbent per trial (step-held), plus re-eval count.

    For each best-so-far changepoint, uses the stored ``test_metric`` user-attr
    when present (future ``log_test_trajectory`` runs), otherwise re-evaluates
    that incumbent's params once via :func:`final_eval` (bounded re-eval — one
    call per distinct incumbent, never per trial). The returned curve holds each
    incumbent's value until the next changepoint.

    :param study: Loaded study (or duck-typed stand-in).
    :param bundle: The dataset bundle to re-evaluate on.
    :param mode: Flavor mode (``split``/``mixed``/``alternate``).
    :param residual: Whether the flavor is residual.
    :param backend: Backend name passed to :func:`final_eval`.
    :param lower: Whether lower objective is better.
    :param n_trials: Total completed-trial count (curve length).
    :param seeds: Final-eval seeds (match the base run's per-dataset count).
    :param embed_layers: Non-monotone embedding depth (base run used 2).
    :returns: ``(curve, n_incumbents_reevaluated)``.
    """
    seeds = list(seeds)
    cps = incumbent_changepoints(study, lower=lower)
    trial_by_index = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    n_eval = 0
    values_at_cp: list[tuple[int, float]] = []
    for idx, params in cps:
        stored = trial_by_index[idx - 1].user_attrs.get("test_metric")
        if stored is not None:
            metric_val = float(stored)
        else:
            agg, _ = final_eval(
                bundle, params, mode=mode, residual=residual, backend=backend,
                seeds=seeds, embed_layers=embed_layers,
            )
            metric_val = float(agg.metric)
            n_eval += 1
        values_at_cp.append((idx, metric_val))
    curve: list[float] = []
    cur = values_at_cp[0][1] if values_at_cp else float("nan")
    j = 0
    for t in range(1, n_trials + 1):
        while j < len(values_at_cp) and values_at_cp[j][0] == t:
            cur = values_at_cp[j][1]
            j += 1
        curve.append(cur)
    return curve, n_eval
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py -q`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/sensitivity_report.py tests/benchmarks/test_sensitivity_report.py
git commit --no-gpg-sign -m "bench(sensitivity): bounded incumbent test-of-incumbent curve"
```

---

### Task 4: Faceted figure (Curve A + Curve B)

**Files:**
- Modify: `benchmarks/_common/sensitivity_report.py`
- Test: `tests/benchmarks/test_sensitivity_report.py`

**Interfaces:**
- Consumes: `best_so_far` (Task 1).
- Produces: `render_plot(series: dict[str, dict[str, tuple[list[float], list[float] | None]]], out_path: Path) -> None`, where `series[dataset][flavor] = (objective_bestsofar, test_curve_or_None)`. Writes `.png` and `.pdf`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/benchmarks/test_sensitivity_report.py
def test_render_plot_writes_png_and_pdf(tmp_path):
    series = {
        "heart": {
            "split-plain": ([0.9, 0.91, 0.91], [0.88, 0.89, 0.89]),
            "alternate-plain": ([0.89, 0.90, 0.906], None),
        }
    }
    out = tmp_path / "sensitivity"
    sr.render_plot(series, out)
    assert (out.with_suffix(".png")).stat().st_size > 0
    assert (out.with_suffix(".pdf")).stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py::test_render_plot_writes_png_and_pdf -q`
Expected: FAIL — `AttributeError: ... 'render_plot'`.

- [ ] **Step 3: Write the implementation**

```python
# add to benchmarks/_common/sensitivity_report.py

def render_plot(
    series: "dict[str, dict[str, tuple[list[float], list[float] | None]]]",
    out_path: "Path",
) -> None:
    r"""Render the sensitivity figure next to ``out_path`` as PNG and PDF.

    One column per dataset; top row = Curve A (best-so-far objective vs trial),
    bottom row = Curve B (test metric of running incumbent vs trial). One line
    per flavor. Agg backend, mathtext labels, no title — supply the docs heading
    / LaTeX caption instead. Mirrors
    :func:`benchmarks._common.size_ladder_report.render_plot`.

    :param series: ``series[dataset][flavor] = (objective_bestsofar, test_curve)``;
        ``test_curve`` may be ``None`` when Curve B was not reconstructed.
    :param out_path: Base output path; the suffix is replaced with png/pdf.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = list(series)
    ncol = max(1, len(datasets))
    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 5.0), squeeze=False)
    for c, ds in enumerate(datasets):
        top, bot = axes[0][c], axes[1][c]
        for fl, (obj, test) in series[ds].items():
            xs = range(1, len(obj) + 1)
            top.plot(xs, obj, marker="", lw=1.5, label=fl)
            if test is not None:
                bot.plot(range(1, len(test) + 1), test, marker="", lw=1.5, label=fl)
        top.set_title(ds, fontsize=11)
        top.set_ylabel("best CV objective", fontsize=10)
        bot.set_ylabel("test of incumbent", fontsize=10)
        bot.set_xlabel(r"Optuna trial $t$", fontsize=10)
        if c == 0:
            top.legend(fontsize=7, loc="best")
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=150, bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_report.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/sensitivity_report.py tests/benchmarks/test_sensitivity_report.py
git commit --no-gpg-sign -m "bench(sensitivity): faceted PNG+PDF figure"
```

---

### Task 5: CLI driver + saturation table

**Files:**
- Create: `benchmarks/sensitivity.py`
- Test: `tests/benchmarks/test_sensitivity_cli.py`

**Interfaces:**
- Consumes: everything in `sensitivity_report`; `benchmarks._common.bundle` for bundles; `benchmarks._common.search._primary_metric`, `_lower_is_better`.
- Produces: `saturation_table(rows: list[dict]) -> str` (Markdown), and a `main()` argparse entry (`--storage-dir`, `--datasets`, `--flavors`, `--out`, `--no-test-curve`).

- [ ] **Step 1: Write the failing test** (table formatting only — no DB access)

```python
# tests/benchmarks/test_sensitivity_cli.py
import pytest
pytest.importorskip("optuna")
from benchmarks.sensitivity import saturation_table


def test_saturation_table_is_markdown_with_medal_columns():
    rows = [
        {"dataset": "heart", "flavor": "split-plain", "trials": 200,
         "t_star": 40, "saturated": True, "n_reeval": 5},
        {"dataset": "loan", "flavor": "mixed-plain", "trials": 50,
         "t_star": 49, "saturated": False, "n_reeval": 3},
    ]
    md = saturation_table(rows)
    assert md.startswith("| dataset | flavor | trials | t*(0.99) |")
    assert "| heart | split-plain | 200 | 40 | ✅ | 5 |" in md
    assert "| loan | mixed-plain | 50 | 49 | ⚠️ | 3 |" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_cli.py -q`
Expected: FAIL — `ModuleNotFoundError: benchmarks.sensitivity`.

- [ ] **Step 3: Write the implementation**

```python
# benchmarks/sensitivity.py
"""CLI: reconstruct HP-search sensitivity curves from committed Optuna storage.

Reconstructs Curve A (best-so-far objective) and, unless ``--no-test-curve``,
Curve B (test-of-incumbent, bounded re-eval) for each ``(dataset, flavor)``
study under ``--storage-dir``; writes the faceted figure to ``--out`` and prints
a Markdown saturation table. Reads storage only — it never re-runs the search.

    uv run --group bench python -m benchmarks.sensitivity \
        --storage-dir benchmarks/results/alternate-base/studies \
        --out docs/_static/hp-search-sensitivity
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from benchmarks.datasets.download import default_dest
from benchmarks.datasets.registry import load  # load(dataset, data_dir) -> DatasetBundle
from benchmarks._common.search import _lower_is_better, _primary_metric
from benchmarks._common.sensitivity_report import (
    best_so_far,
    completed_values,
    incumbent_test_curve,
    load_study,
    render_plot,
    saturation_trial,
)

# Final-eval seed count per dataset used by the base run (match for comparable
# test metrics): small/medium 20, large single-holdout 10.
_FINAL_SEEDS = {"auto": 20, "heart": 20, "compas": 10, "blog": 10, "loan": 10}
_ORDER = ["heart", "auto", "compas", "blog", "loan"]


def _mode_residual(flavor: str) -> tuple[str, bool]:
    """Split a ``{mode}-{plain|residual}`` flavor label into ``(mode, residual)``.

    ``mixed-fixed-*`` maps to ``mode="mixed"`` (convex_fraction is a param, not a
    mode); its stored params omit ``convex_fraction``, so ``final_eval`` defaults
    it to 0.5 — exactly the fixed arm.

    :param flavor: Study flavor label, e.g. ``mixed-fixed-plain``.
    :returns: ``(mode, residual)``.
    """
    residual = "residual" in flavor or "deep" in flavor
    mode = "mixed" if flavor.startswith("mixed") else flavor.split("-")[0]
    return mode, residual


def saturation_table(rows: list[dict[str, Any]]) -> str:
    """Render the per-study saturation summary as a GitHub-flavored Markdown table.

    :param rows: One dict per study with keys ``dataset, flavor, trials, t_star,
        saturated, n_reeval``.
    :returns: The Markdown table as a single string.
    """
    out = [
        "| dataset | flavor | trials | t*(0.99) | saturated | # re-eval |",
        "|---|---|--:|--:|:-:|--:|",
    ]
    for r in rows:
        mark = "✅" if r["saturated"] else "⚠️"
        out.append(
            f"| {r['dataset']} | {r['flavor']} | {r['trials']} | "
            f"{r['t_star']} | {mark} | {r['n_reeval']} |"
        )
    return "\n".join(out)


def main() -> None:
    """Reconstruct curves for each study under ``--storage-dir`` and emit outputs."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--storage-dir", required=True)
    ap.add_argument("--datasets", nargs="*", default=_ORDER)
    ap.add_argument("--flavors", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-test-curve", action="store_true")
    args = ap.parse_args()

    store = Path(args.storage_dir)
    series: dict[str, dict[str, tuple[list[float], list[float] | None]]] = {}
    table_rows: list[dict[str, Any]] = []
    for ds in args.datasets:
        bundle = load(ds, data_dir=default_dest())
        metric = _primary_metric(bundle)
        lower = _lower_is_better(metric)
        dbs = sorted(store.glob(f"{ds}-*.db"))
        series[ds] = {}
        for db in dbs:
            flavor = db.stem[len(ds) + 1 :]
            if args.flavors and flavor not in args.flavors:
                continue
            study = load_study(db, db.stem)
            vals = completed_values(study, lower)
            if not vals:
                continue
            obj = best_so_far(vals, lower)
            t_star = saturation_trial(obj, lower)
            test_curve: list[float] | None = None
            n_reeval = 0
            if not args.no_test_curve:
                mode, residual = _mode_residual(flavor)
                test_curve, n_reeval = incumbent_test_curve(
                    study, bundle, mode=mode, residual=residual, backend="torch",
                    lower=lower, n_trials=len(vals),
                    seeds=range(_FINAL_SEEDS.get(ds, 10)),
                )
            series[ds][flavor] = (obj, test_curve)
            table_rows.append({
                "dataset": ds, "flavor": flavor, "trials": len(vals),
                "t_star": t_star, "saturated": t_star < len(vals),
                "n_reeval": n_reeval,
            })
    render_plot(series, Path(args.out))
    print(saturation_table(table_rows))  # noqa: T201


if __name__ == "__main__":
    main()
```

> **Note:** bundle loading mirrors `search.run_dataset` exactly — `load(dataset, data_dir=default_dest())` from `benchmarks.datasets.registry` / `benchmarks.datasets.download`. Confirmed against the merged tree.

- [ ] **Step 4: Run test to verify it passes**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_sensitivity_cli.py -q`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/sensitivity.py tests/benchmarks/test_sensitivity_cli.py
git commit --no-gpg-sign -m "bench(sensitivity): CLI driver + saturation table"
```

---

### Task 6: Prospective per-trial test logging (optional flag)

**Files:**
- Modify: `benchmarks/_common/search.py` (the `search()` function's `objective`)
- Test: `tests/benchmarks/test_search_log_test_trajectory.py`

**Interfaces:**
- Consumes: `search()` (merged #111 signature).
- Produces: `search(..., log_test_trajectory: bool = False)`; when set, each trial records `trial.set_user_attr("test_metric", <held-out test metric>)`. Never read back into the objective value.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_search_log_test_trajectory.py
import pytest
pytest.importorskip("optuna")
pytest.importorskip("torch")
from benchmarks._common.bundle import load_bundle
from benchmarks._common.search import search


@pytest.mark.slow
def test_log_test_trajectory_sets_user_attr(tmp_path):
    import optuna
    from benchmarks.datasets.download import default_dest
    from benchmarks.datasets.registry import load

    bundle = load("heart", data_dir=default_dest())
    storage = f"sqlite:///{tmp_path}/heart-split-plain.db"
    res = search(
        bundle, mode="split", residual=False, backend="torch",
        n_trials=2, n_splits=2, search_seeds=1, epochs=2,
        embed_layers=2, storage=storage, log_test_trajectory=True,
    )
    assert res.n_trials == 2
    study = optuna.load_study(study_name="heart-split-plain", storage=storage)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert done and all(
        isinstance(t.user_attrs.get("test_metric"), float) for t in done
    )
```

`search()` returns `StudyResult`, not the study, so the test reloads the study
from the `storage` path it passed — no change to `StudyResult` needed. The
study name is `{dataset}-{flavor}` = `heart-split-plain` (from
`_run_flavor_label`).

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_search_log_test_trajectory.py -q -m slow`
Expected: FAIL — `TypeError: search() got an unexpected keyword argument 'log_test_trajectory'`.

- [ ] **Step 3: Write the implementation**

Add the parameter to `search()` and, inside `objective`, after computing the CV `scores` (do NOT alter the returned objective), when `log_test_trajectory` is set, evaluate the trial's config on the held-out test split once and store it:

```python
def search(
    bundle: DatasetBundle,
    *,
    ...,
    search_convex_fraction: bool = True,
    log_test_trajectory: bool = False,   # NEW
) -> StudyResult:
    ...
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest_config(...)
        cfg = dataclasses.replace(cfg, seeds=tuple(range(search_seeds)))
        scores = [...]  # unchanged CV loop
        obj = float(arr.mean() + arr.std()) if lower else float(arr.mean() - arr.std())
        if log_test_trajectory:
            # Diagnostic only — never fed back into `obj`.
            agg, _ = final_eval(
                bundle, dict(trial.params), mode=mode, residual=residual,
                backend=backend, seeds=range(1), embed_layers=embed_layers,
            )
            trial.set_user_attr("test_metric", float(agg.metric))
        return obj
```

Document the flag in the `search` docstring (`:param log_test_trajectory:`), noting it roughly doubles per-trial cost and is a pure diagnostic.

- [ ] **Step 4: Run test to verify it passes**

Run: `MONONET_TORCH_DEVICE=cpu uv run --no-sync pytest tests/benchmarks/test_search_log_test_trajectory.py -q -m slow`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/search.py tests/benchmarks/test_search_log_test_trajectory.py
git commit --no-gpg-sign -m "bench(sensitivity): optional per-trial test_metric logging"
```

---

### Task 7: Docs page + committed figures + wiring

**Files:**
- Create: `docs/benchmarks/hp-search-sensitivity.md`
- Create: `docs/_static/hp-search-sensitivity.png` and `.pdf` (generated)
- Modify: the benchmarks docs toctree (`docs/benchmarks/index.md` or the parent that lists `alternate-base-result.md`)

**Interfaces:**
- Consumes: the Task-5 CLI.

- [ ] **Step 1: Generate the committed figure + table** (requires #111's DBs present)

Run:
```bash
MONONET_TORCH_DEVICE=cpu uv run --group bench python -m benchmarks.sensitivity \
  --storage-dir benchmarks/results/alternate-base/studies \
  --out docs/_static/hp-search-sensitivity > /tmp/sat-table.md
```
Expected: writes `docs/_static/hp-search-sensitivity.{png,pdf}` and a saturation table on stdout.

- [ ] **Step 2: Write the docs page**

Create `docs/benchmarks/hp-search-sensitivity.md` with: a one-paragraph intro (why budget-sensitivity matters, linking the base-result page), the embedded figure (`![](../_static/hp-search-sensitivity.png)`), the saturation table (paste from `/tmp/sat-table.md`), and an interpretation section that calls out (a) which flavors saturated within budget and which were still climbing, and (b) `auto`'s meta-overfitting (Curve B turning back up while Curve A keeps improving). Note the `# re-eval` column is the bounded incumbent-reeval count (no search re-run).

- [ ] **Step 3: Wire into the toctree**

Add `hp-search-sensitivity` to the same toctree/index that lists `alternate-base-result` (match the existing entry's style).

- [ ] **Step 4: Build the docs to verify**

Run: `./tools/build-docs.sh`
Expected: builds without warnings about the new page; the figure renders.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/hp-search-sensitivity.md docs/_static/hp-search-sensitivity.png docs/_static/hp-search-sensitivity.pdf docs/benchmarks/index.md
git commit --no-gpg-sign -m "docs(sensitivity): HP-search sensitivity page + committed figure"
```

---

## Self-review notes

- **Spec coverage:** Curve A + saturation (Tasks 1–2), Curve B bounded re-eval (Task 3), faceted figure (Task 4), CLI + saturation table (Task 5), prospective logging hook (Task 6), docs + committed figures (Task 7). Bands/random-order left as spec follow-ups (not tasks), matching the spec's scope-out.
- **Prerequisite gate:** every task that touches storage or `final_eval` depends on #111 being merged; Global Constraints state this and Tasks 5/7 note the two interface confirmations (`load_bundle` name, `search` study exposure) that only the merged tree can pin.
- **Type consistency:** `best_so_far`/`saturation_trial`/`incumbent_changepoints`/`incumbent_test_curve`/`render_plot` names are used identically in the CLI (Task 5) and tests. `series[ds][flavor] = (obj, test|None)` shape is consistent between Task 4 and Task 5.
