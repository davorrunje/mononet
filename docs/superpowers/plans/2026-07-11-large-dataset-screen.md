# Large-dataset monotonic-depth screen (Phase 1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the data plumbing (LFS + git-lfs devcontainers), the max-size deep/shallow screen, the ladder-or-standard gate, and the report/docs — proven end-to-end on one reference dataset (Adult) — so the remaining roster datasets are a mechanical repeat.

**Architecture:** Reuse the existing `benchmarks/` harness. The screen is `run_ladder` (already bundle-generic) run at a single full-size rung for the deep and shallow `absolute`-residual arms; `delta_by_n` gives Δ ± bootstrap CI; a pure `gate()` routes each dataset to `ladder` or `standard`. Datasets are `DatasetSpec` entries loaded from CSV by the existing loader; redistributable data is committed gzip-compressed to Git LFS, restricted data ships a prep script.

**Tech Stack:** Python 3.11+, numpy, pandas (bench group), matplotlib, Optuna, PyTorch, Sphinx+myst-nb, Git LFS, uv.

## Global Constraints

- Python 3.11+, ruff line length 88; strict mypy; type hints on every function.
- MyST field-list docstrings (`:param:`/`:returns:`/`:raises:`) on all public functions/classes; no `:type:`/`:rtype:`.
- Stdlib `dataclasses` only — **no Pydantic**.
- Benchmark-only: **no change to `mononet/`** (package/kernel/model_builder). `benchmarks/` stays out of the wheel.
- Protocol: the **test split is full and never subsampled**; selection is train-only CV; final numbers are multi-seed test IQM.
- Hosting: LFS/committed data **only** for redistributable licenses (Zenodo/UCI/CC/public-domain); Kaggle-restricted → prep script, nothing committed.
- All roster datasets are **binary classification** (accuracy, higher-is-better); no regression path is implemented in Phase 1.
- Backends selected per existing convention; screen runs on `backend="torch"`.

---

## File structure

- Create `benchmarks/datasets/sources.py` — hosting-class descriptor + `require_dataset()` resolver/error.
- Create `benchmarks/_common/screen_gate.py` — pure `gate()`.
- Create `benchmarks/large_screen_run.py` — `screen_dataset()` + CLI.
- Create `benchmarks/_common/screen_report.py` — `screen_table()` + `render_screen_plot()`.
- Create `benchmarks/datasets/prepare/__init__.py`, `benchmarks/datasets/prepare/adult.py` — raw→CSV prep for the reference dataset.
- Create `benchmarks/data/adult/{train,test}_adult.csv.gz` — committed LFS data.
- Create `docs/benchmarks/large-dataset-screen.md`, `benchmarks/RUNBOOK-large-screen.md`.
- Modify `.gitattributes` (LFS pattern), `.devcontainer/shared/install_common_tools.sh` (git-lfs), `benchmarks/datasets/spec.py` (+adult spec), `benchmarks/_common/search.py` (`_BUDGET` +adult), `docs/benchmarks/index.md` (toctree).
- Tests under `tests/benchmarks/`.

---

### Task 1: LFS plumbing + git-lfs in devcontainers

**Files:**
- Modify: `.gitattributes`
- Modify: `.devcontainer/shared/install_common_tools.sh`
- Test: `tests/benchmarks/test_lfs_layout.py`

**Interfaces:**
- Produces: the `benchmarks/data/**` LFS filter rule that Task 7 relies on to commit `.csv.gz`.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_lfs_layout.py
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_data_dir_is_lfs_tracked() -> None:
    """Any file under benchmarks/data/ resolves to the git-lfs filter."""
    out = subprocess.run(
        ["git", "check-attr", "filter", "--", "benchmarks/data/adult/train_adult.csv.gz"],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout
    assert "filter: lfs" in out


def test_devcontainer_installs_git_lfs() -> None:
    """The shared devcontainer setup installs and initialises git-lfs."""
    script = (REPO / ".devcontainer/shared/install_common_tools.sh").read_text()
    assert "git-lfs" in script
    assert "git lfs install" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_lfs_layout.py -v`
Expected: FAIL (`filter: lfs` absent; script lacks git-lfs).

- [ ] **Step 3: Add the LFS pattern to `.gitattributes`**

Append:

```gitattributes
# Benchmark datasets (redistributable only) — stored via Git LFS
benchmarks/data/** filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 4: Install git-lfs in the shared devcontainer setup**

In `.devcontainer/shared/install_common_tools.sh`, add near the other apt installs (adapt to the script's existing install idiom):

```bash
# git-lfs — required to pull committed benchmark datasets under benchmarks/data/
if ! command -v git-lfs >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y git-lfs
fi
git lfs install --skip-repo
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_lfs_layout.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitattributes .devcontainer/shared/install_common_tools.sh tests/benchmarks/test_lfs_layout.py
git commit -m "bench: git-lfs devcontainer support + benchmarks/data LFS filter"
```

---

### Task 2: Data-source descriptor + loader resolver

**Files:**
- Create: `benchmarks/datasets/sources.py`
- Test: `tests/benchmarks/test_sources.py`

**Interfaces:**
- Produces:
  - `DataSource` dataclass: `name: str`, `hosting: Literal["lfs", "script"]`, `license: str`, `url: str`, `prep_hint: str`.
  - `SOURCES: dict[str, DataSource]`.
  - `resolve_dir(name: str) -> Path` — returns the committed dir `benchmarks/data/<name>` for `lfs`, else the local cache `default_dest()`.
  - `require_dataset(name: str) -> Path` — returns the resolved dir if the expected `train_<name>.csv[.gz]` exists there; otherwise raises `FileNotFoundError` with an actionable message (Task 5 + Task 7 consume this).

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_sources.py
from __future__ import annotations

import pytest

from benchmarks.datasets.sources import SOURCES, DataSource, require_dataset


def test_adult_is_lfs_hosted() -> None:
    src = SOURCES["adult"]
    assert isinstance(src, DataSource)
    assert src.hosting == "lfs"
    assert "uci" in src.url.lower() or "openml" in src.url.lower()


def test_require_dataset_missing_script_source_raises_actionable(tmp_path, monkeypatch) -> None:
    """A script-only dataset with no local file raises with prep instructions."""
    monkeypatch.setenv("MONONET_DATA_DIR", str(tmp_path))
    SOURCES["_fake_script"] = DataSource(
        name="_fake_script", hosting="script", license="Kaggle ToS",
        url="https://kaggle.com/x", prep_hint="run prepare/_fake_script.py",
    )
    with pytest.raises(FileNotFoundError, match="prepare/_fake_script.py"):
        require_dataset("_fake_script")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_sources.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `sources.py`**

```python
"""Per-dataset hosting descriptor and local-file resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from benchmarks.datasets.download import default_dest

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


@dataclass(frozen=True, slots=True)
class DataSource:
    """Where a dataset's preprocessed CSVs come from.

    :param hosting: ``lfs`` (committed under ``benchmarks/data/<name>/``) or
        ``script`` (user regenerates into the local cache from restricted raw).
    :param prep_hint: One-line instruction shown when a script-only dataset is
        missing locally.
    """

    name: str
    hosting: Literal["lfs", "script"]
    license: str
    url: str
    prep_hint: str


SOURCES: dict[str, DataSource] = {
    "adult": DataSource(
        name="adult",
        hosting="lfs",
        license="CC-BY-4.0 (UCI)",
        url="https://archive.ics.uci.edu/dataset/2/adult",
        prep_hint="committed via LFS; regenerate with prepare/adult.py",
    ),
}


def resolve_dir(name: str) -> Path:
    """Directory that should contain ``train_<name>.csv[.gz]`` for *name*."""
    src = SOURCES[name]
    return _DATA_ROOT / name if src.hosting == "lfs" else default_dest()


def require_dataset(name: str) -> Path:
    """Resolve *name*'s data dir, asserting the train file is present.

    :raises FileNotFoundError: If no ``train_<name>.csv[.gz]`` is found, with a
        message pointing at the prep step / ``git lfs pull``.
    """
    d = resolve_dir(name)
    if any((d / f"train_{name}{ext}").exists() for ext in (".csv", ".csv.gz")):
        return d
    src = SOURCES[name]
    raise FileNotFoundError(
        f"Dataset {name!r} not found in {d}. Source: {src.url} ({src.license}). "
        f"{'Run `git lfs pull`.' if src.hosting == 'lfs' else src.prep_hint}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/benchmarks/test_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/datasets/sources.py tests/benchmarks/test_sources.py
git commit -m "bench: dataset hosting descriptor + require_dataset resolver"
```

---

### Task 3: `.csv.gz` support in the loader

**Files:**
- Modify: `benchmarks/datasets/loader.py:18-25` (the `_read_csv` open path)
- Test: `tests/benchmarks/test_loader_gz.py`

**Interfaces:**
- Consumes: `load_spec(spec, *, data_dir)` (existing).
- Produces: `_read_csv` transparently reads `.csv` or `.csv.gz`; `load_spec` finds either. Task 7 relies on gz loading.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_loader_gz.py
from __future__ import annotations

import gzip
from pathlib import Path

from benchmarks.datasets.loader import load_spec
from benchmarks.datasets.spec import DatasetSpec


def _write_gz(p: Path, text: str) -> None:
    with gzip.open(p, "wt", encoding="utf-8") as fh:
        fh.write(text)


def test_load_spec_reads_gzip(tmp_path: Path) -> None:
    rows = "f0,f1,ground_truth\n0.1,0.2,1\n0.3,0.4,0\n"
    _write_gz(tmp_path / "train_toy.csv.gz", rows)
    _write_gz(tmp_path / "test_toy.csv.gz", rows)
    spec = DatasetSpec("toy", "binary_classification", "ground_truth", ("f0",), ("f1",))
    b = load_spec(spec, data_dir=tmp_path)
    assert b.X_train.shape == (2, 2)
    assert b.mono_increasing == (0,) and b.mono_decreasing == (1,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_loader_gz.py -v`
Expected: FAIL (only `.csv` is opened).

- [ ] **Step 3: Make `_read_csv` + path resolution gz-aware**

Replace the body of `_read_csv` and add a resolver. New `_read_csv`:

```python
import gzip
import io


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a ``.csv`` or ``.csv.gz`` file into a header and float64 array."""
    if path.suffix == ".gz":
        fh: io.TextIOBase = gzip.open(path, "rt", newline="", encoding="utf-8")
    else:
        fh = path.open(newline="", encoding="utf-8")
    with fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    return header, np.array(rows, dtype=np.float64)


def _find(data_dir: Path, stem: str) -> Path:
    """Return ``<stem>.csv`` or ``<stem>.csv.gz`` under *data_dir*."""
    for ext in (".csv", ".csv.gz"):
        p = data_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"{stem}.csv[.gz] not found in {data_dir}")
```

In `load_spec`, change the two reads to use `_find`:

```python
    header, train = _read_csv(_find(data_dir, f"train_{spec.name}"))
    _, test = _read_csv(_find(data_dir, f"test_{spec.name}"))
```

- [ ] **Step 4: Run test + existing loader tests**

Run: `uv run pytest tests/benchmarks/test_loader_gz.py -v && uv run mypy benchmarks/datasets/loader.py`
Expected: PASS; mypy clean.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/datasets/loader.py tests/benchmarks/test_loader_gz.py
git commit -m "bench: loader reads .csv.gz transparently"
```

---

### Task 4: The gate

**Files:**
- Create: `benchmarks/_common/screen_gate.py`
- Test: `tests/benchmarks/test_screen_gate.py`

**Interfaces:**
- Produces: `gate(delta_lo: float, delta_point: float, margin: float) -> Literal["ladder", "standard"]` and `DEFAULT_MARGIN = 0.005`. Task 5 consumes both.

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_screen_gate.py
from __future__ import annotations

import pytest

from benchmarks._common.screen_gate import DEFAULT_MARGIN, gate


@pytest.mark.parametrize(
    ("lo", "point", "expect"),
    [
        (0.001, 0.010, "ladder"),    # CI clears 0 and point clears margin
        (0.001, 0.004, "standard"),  # CI clears 0 but point below margin
        (-0.001, 0.010, "standard"), # point big but CI touches 0
        (0.0, 0.010, "standard"),    # lo == 0 is not > 0
        (0.006, 0.005, "ladder"),    # point == margin qualifies
    ],
)
def test_gate_boundaries(lo: float, point: float, expect: str) -> None:
    assert gate(lo, point, DEFAULT_MARGIN) == expect
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_screen_gate.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `screen_gate.py`**

```python
"""Screen verdict gate: does a dataset advance to a full ladder study?"""

from __future__ import annotations

from typing import Literal

DEFAULT_MARGIN = 0.005
"""Practical-significance floor on Δ for classification (accuracy)."""


def gate(
    delta_lo: float, delta_point: float, margin: float = DEFAULT_MARGIN
) -> Literal["ladder", "standard"]:
    """Route a dataset by its max-size deep−shallow gap.

    :param delta_lo: Lower bound of the 95% seed-bootstrap band on Δ.
    :param delta_point: Point estimate of Δ = IQM(deep) − IQM(shallow).
    :param margin: Practical-significance floor.
    :returns: ``ladder`` iff Δ is significantly (``delta_lo > 0``) *and*
        practically (``delta_point >= margin``) positive; else ``standard``.
    """
    if delta_lo > 0.0 and delta_point >= margin:
        return "ladder"
    return "standard"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/benchmarks/test_screen_gate.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/screen_gate.py tests/benchmarks/test_screen_gate.py
git commit -m "bench: add CI+practical-floor screen gate"
```

---

### Task 5: Max-size screen runner

**Files:**
- Create: `benchmarks/large_screen_run.py`
- Modify: `benchmarks/_common/search.py` (`_BUDGET` — add `adult`)
- Test: `tests/benchmarks/test_large_screen_run.py`

**Interfaces:**
- Consumes: `run_ladder` (from `benchmarks.loan_size_ladder_run`), `delta_by_n` (`benchmarks._common.size_ladder_report`), `gate`/`DEFAULT_MARGIN` (Task 4), `require_dataset` (Task 2).
- Produces: `screen_dataset(bundle, *, n_trials, search_seeds, final_seeds, epochs, n_jobs, backend="torch") -> dict` returning `{name, n_full, deep_iqm, shallow_iqm, delta, delta_lo, delta_hi, margin, verdict}`; and a `main()` CLI (`--dataset`, budget flags, `--out`).

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_large_screen_run.py
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks.large_screen_run import screen_dataset


def _toy_bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 3))
    y = (x[:, 0] + x[:, 1] - x[:, 2] > 0).astype(float)
    xt = rng.normal(size=(80, 3))
    yt = (xt[:, 0] + xt[:, 1] - xt[:, 2] > 0).astype(float)
    return DatasetBundle(
        name="toy", task="binary_classification",
        X_train=x, y_train=y, X_test=xt, y_test=yt,
        mono_increasing=(0, 1), mono_decreasing=(2,),
        feature_names=("f0", "f1", "f2"), metadata={},
    )


def test_screen_dataset_smoke() -> None:
    """Tiny budget end-to-end: a record with finite Δ and a valid verdict."""
    rec = screen_dataset(
        _toy_bundle(), n_trials=1, search_seeds=1, final_seeds=2, epochs=1, n_jobs=1
    )
    assert rec["name"] == "toy"
    assert np.isfinite(rec["delta"])
    assert rec["verdict"] in {"ladder", "standard"}
    assert rec["delta_lo"] <= rec["delta"] <= rec["delta_hi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_large_screen_run.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `large_screen_run.py`**

```python
"""Max-size deep/shallow screen: one Δ + gate verdict per dataset.

Runs the standard search for the deep and shallow ``absolute``-residual arms at
the dataset's full train size, multi-seed refit + test, and gates on
Δ = IQM(deep) − IQM(shallow). See
docs/superpowers/specs/2026-07-11-large-dataset-screen-design.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks._common.screen_gate import DEFAULT_MARGIN, gate
from benchmarks._common.size_ladder_report import delta_by_n
from benchmarks.loan_size_ladder_run import run_ladder

if TYPE_CHECKING:
    from benchmarks._common.bundle import DatasetBundle

_FULL = 1_000_000_000  # >= any train size ⇒ subsample_train returns the full split


def screen_dataset(
    bundle: DatasetBundle,
    *,
    n_trials: int = 25,
    search_seeds: int = 3,
    final_seeds: int = 10,
    epochs: int = 50,
    n_jobs: int = 1,
    backend: str = "torch",
    margin: float = DEFAULT_MARGIN,
) -> dict[str, Any]:
    """Screen one dataset at full size; return the record + gate verdict."""
    recs = run_ladder(
        bundle, ns=(_FULL,), arms=("deep", "shallow"), backend=backend,
        n_trials=n_trials, search_seeds=search_seeds,
        final_seeds=range(final_seeds), epochs=epochs, n_jobs=n_jobs,
    )
    d = delta_by_n(recs)[0]
    return {
        "name": bundle.name,
        "n_full": d["n"],
        "deep_iqm": d["deep_iqm"],
        "shallow_iqm": d["shallow_iqm"],
        "delta": d["delta"],
        "delta_lo": d["delta_lo"],
        "delta_hi": d["delta_hi"],
        "margin": margin,
        "verdict": gate(d["delta_lo"], d["delta"], margin),
    }


def main() -> None:
    """CLI: screen one dataset and write its record JSON."""
    import argparse

    from benchmarks.datasets.registry import load
    from benchmarks.datasets.sources import require_dataset

    ap = argparse.ArgumentParser(description="max-size deep/shallow screen")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--search-seeds", type=int, default=3)
    ap.add_argument("--final-seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bundle = load(args.dataset, data_dir=require_dataset(args.dataset))
    rec = screen_dataset(
        bundle, n_trials=args.n_trials, search_seeds=args.search_seeds,
        final_seeds=args.final_seeds, epochs=args.epochs, n_jobs=args.n_jobs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"{args.dataset}: Δ={rec['delta']:+.4f} verdict={rec['verdict']}")  # noqa: T201


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the `adult` budget entry to `_BUDGET`**

In `benchmarks/_common/search.py`, add to `_BUDGET`:

```python
    "adult": (25, range(10), 5),
```

- [ ] **Step 5: Run test + mypy**

Run: `uv run pytest tests/benchmarks/test_large_screen_run.py -v && uv run mypy benchmarks/large_screen_run.py`
Expected: PASS; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add benchmarks/large_screen_run.py benchmarks/_common/search.py tests/benchmarks/test_large_screen_run.py
git commit -m "bench: max-size deep/shallow screen runner + gate wiring"
```

---

### Task 6: Screen report (table + plot)

**Files:**
- Create: `benchmarks/_common/screen_report.py`
- Test: `tests/benchmarks/test_screen_report.py`

**Interfaces:**
- Consumes: screen records (Task 5 schema).
- Produces: `screen_table(records: list[dict]) -> str` (Markdown), `render_screen_plot(records: list[dict], out_path: Path) -> None` (writes `.png` + `.pdf`, sorted by Δ, reference lines at 0 and the margin).

- [ ] **Step 1: Write the failing test**

```python
# tests/benchmarks/test_screen_report.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmarks._common.screen_report import screen_table


def _rec(name: str, delta: float, lo: float, hi: float, verdict: str) -> dict[str, Any]:
    return {
        "name": name, "n_full": 1000, "deep_iqm": 0.7, "shallow_iqm": 0.7 - delta,
        "delta": delta, "delta_lo": lo, "delta_hi": hi, "margin": 0.005,
        "verdict": verdict,
    }


def test_screen_table_has_row_per_dataset_and_verdict() -> None:
    rows = [_rec("a", 0.01, 0.006, 0.014, "ladder"), _rec("b", 0.0, -0.003, 0.003, "standard")]
    md = screen_table(rows)
    assert "| a |" in md and "| b |" in md
    assert "ladder" in md and "standard" in md


def test_render_screen_plot_writes_png_and_pdf(tmp_path: Path) -> None:
    import pytest
    pytest.importorskip("matplotlib")
    from benchmarks._common.screen_report import render_screen_plot

    rows = [_rec("a", 0.01, 0.006, 0.014, "ladder"), _rec("b", 0.0, -0.003, 0.003, "standard")]
    render_screen_plot(rows, tmp_path / "screen.png")
    for suffix in (".png", ".pdf"):
        out = (tmp_path / "screen").with_suffix(suffix)
        assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_screen_report.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `screen_report.py`**

```python
"""Render the max-size screen as a Markdown table + a Δ-per-dataset plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


def screen_table(records: list[dict[str, Any]]) -> str:
    """Markdown table: one row per dataset with Δ ± CI and the verdict."""
    head = (
        "| dataset | N | deep IQM | shallow IQM | Δ [95% CI] | verdict |\n"
        "|---|--:|--:|--:|--:|:--|\n"
    )
    rows = [
        f"| {r['name']} | {r['n_full']} | {r['deep_iqm']:.4f} | "
        f"{r['shallow_iqm']:.4f} | {r['delta']:+.4f} "
        f"[{r['delta_lo']:+.4f}, {r['delta_hi']:+.4f}] | {r['verdict']} |"
        for r in records
    ]
    return head + "\n".join(rows) + "\n"


def render_screen_plot(records: list[dict[str, Any]], out_path: Path) -> None:
    """Δ per dataset (sorted) with CI bars + 0 and margin reference lines.

    Writes both ``out_path`` (``.png``) and a sibling ``.pdf`` (vector, LaTeX).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = sorted(records, key=lambda r: r["delta"])
    names = [r["name"] for r in recs]
    delta = np.array([r["delta"] for r in recs])
    lo = np.clip(delta - np.array([r["delta_lo"] for r in recs]), 0.0, None)
    hi = np.clip(np.array([r["delta_hi"] for r in recs]) - delta, 0.0, None)
    margin = recs[0]["margin"] if recs else 0.005

    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(recs) + 1.5))
    y = np.arange(len(recs))
    ax.errorbar(delta, y, xerr=[lo, hi], fmt="o", color="#0072B2", capsize=3)
    ax.axvline(0.0, color="0.4", lw=1)
    ax.axvline(margin, color="0.7", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(r"$\Delta$ = IQM(deep) − IQM(shallow)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_screen_report.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/screen_report.py tests/benchmarks/test_screen_report.py
git commit -m "bench: screen report table + Δ-per-dataset plot (PNG+PDF)"
```

---

### Task 7: Adult reference dataset (prep + spec + committed LFS data)

**Files:**
- Create: `benchmarks/datasets/prepare/__init__.py`, `benchmarks/datasets/prepare/adult.py`
- Modify: `benchmarks/datasets/spec.py` (add `adult` to `DATASETS_SPEC`)
- Create (committed via LFS): `benchmarks/data/adult/train_adult.csv.gz`, `benchmarks/data/adult/test_adult.csv.gz`
- Test: `tests/benchmarks/test_prepare_adult.py`

**Interfaces:**
- Consumes: gz loader (Task 3), `SOURCES["adult"]` (Task 2).
- Produces: `prepare_adult(raw_df: pandas.DataFrame) -> tuple[pandas.DataFrame, pandas.DataFrame]` (train, test) in mononet convention — target renamed `ground_truth` (0/1), all columns numeric, ordinal monotone columns `education_num`, `hours_per_week`, `capital_gain` preserved; a fixed stratified 80/20 split (seed 0). Registry `load("adult", ...)` returns a bundle with those three as `mono_increasing`.

- [ ] **Step 1: Write the failing test (prep on a synthetic raw sample — no real download)**

```python
# tests/benchmarks/test_prepare_adult.py
from __future__ import annotations

import numpy as np
import pandas as pd

from benchmarks.datasets.prepare.adult import MONO_INCREASING, prepare_adult


def _synthetic_raw(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "age": rng.integers(18, 80, n),
            "education_num": rng.integers(1, 16, n),
            "hours_per_week": rng.integers(1, 80, n),
            "capital_gain": rng.integers(0, 5000, n),
            "workclass": rng.choice(["Private", "Gov"], n),
            "income": rng.choice([">50K", "<=50K"], n),
        }
    )


def test_prepare_adult_contract() -> None:
    train, test = prepare_adult(_synthetic_raw())
    for df in (train, test):
        assert "ground_truth" in df.columns
        assert set(df["ground_truth"].unique()) <= {0, 1}
        assert df.select_dtypes("object").empty  # all numeric
        for col in MONO_INCREASING:
            assert col in df.columns
    assert len(train) == 160 and len(test) == 40  # 80/20 of 200
    assert len(set(train.index) & set(test.index)) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_prepare_adult.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `prepare/adult.py`**

```python
"""Prepare the UCI Adult (Census Income) dataset in mononet convention.

Monotone (increasing) in ``education_num``, ``hours_per_week``, ``capital_gain``;
``SEX``/``RACE`` are deliberately NOT constrained. Categorical columns are
one-hot encoded; the binary income target becomes ``ground_truth`` (1 = >50K).
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

MONO_INCREASING: tuple[str, ...] = ("education_num", "hours_per_week", "capital_gain")


def prepare_adult(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) numeric frames with a fixed stratified 80/20 split.

    :param raw: Raw Adult frame with an ``income`` target column.
    :returns: Train/test frames; target is ``ground_truth`` (0/1), the monotone
        ordinal columns are preserved, categoricals are one-hot encoded.
    """
    df = raw.copy()
    df["ground_truth"] = (df.pop("income").astype(str).str.contains(">50K")).astype(int)
    cat = [c for c in df.columns if df[c].dtype == object and c not in MONO_INCREASING]
    df = pd.get_dummies(df, columns=cat, dtype=int)
    train, test = train_test_split(
        df, test_size=0.2, random_state=0, stratify=df["ground_truth"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
```

- [ ] **Step 4: Run prep test to verify it passes**

Run: `uv run pytest tests/benchmarks/test_prepare_adult.py -v`
Expected: PASS.

- [ ] **Step 5: Register the spec**

Add to `DATASETS_SPEC` in `benchmarks/datasets/spec.py`:

```python
    "adult": DatasetSpec(
        "adult",
        "binary_classification",
        "ground_truth",
        ("education_num", "hours_per_week", "capital_gain"),
        (),
    ),
```

- [ ] **Step 6: Materialise + commit the real data via LFS**

Download the real Adult data and run the prep to produce the committed gz files (one-off, documented in the RUNBOOK):

```bash
uv run --group bench python - <<'PY'
from pathlib import Path
import pandas as pd
from benchmarks.datasets.prepare.adult import prepare_adult
cols = ["age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","income"]
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
raw = pd.read_csv(url, header=None, names=cols, skipinitialspace=True, na_values="?").dropna()
raw = raw.rename(columns={"education-num":"education_num","hours-per-week":"hours_per_week","capital-gain":"capital_gain"})
train, test = prepare_adult(raw)
d = Path("benchmarks/data/adult"); d.mkdir(parents=True, exist_ok=True)
train.to_csv(d/"train_adult.csv.gz", index=False, compression="gzip")
test.to_csv(d/"test_adult.csv.gz", index=False, compression="gzip")
print("wrote", len(train), len(test))
PY
git add benchmarks/data/adult/train_adult.csv.gz benchmarks/data/adult/test_adult.csv.gz
git lfs ls-files | grep adult   # confirm LFS-tracked
```

- [ ] **Step 7: Verify the loader loads the committed data**

Run:

```bash
uv run --group bench python -c "
from benchmarks.datasets.registry import load
from benchmarks.datasets.sources import require_dataset
b = load('adult', data_dir=require_dataset('adult'))
print(b.X_train.shape, b.mono_increasing)
assert len(b.mono_increasing) == 3
"
```

Expected: prints a shape with >30k train rows and `(i, j, k)` indices.

- [ ] **Step 8: Commit**

```bash
git add benchmarks/datasets/prepare/__init__.py benchmarks/datasets/prepare/adult.py \
        benchmarks/datasets/spec.py tests/benchmarks/test_prepare_adult.py
git commit -m "bench: add Adult reference dataset (prep + spec + LFS data)"
```

---

### Task 8: Docs page + RUNBOOK + toctree

**Files:**
- Create: `docs/benchmarks/large-dataset-screen.md`
- Create: `benchmarks/RUNBOOK-large-screen.md`
- Modify: `docs/benchmarks/index.md` (toctree)
- Test: `sphinx-build -W`

**Interfaces:**
- Consumes: `screen_table`/`render_screen_plot` (Task 6). Page embeds `../_static/large-dataset-screen.png` (committed when the real screen runs).

- [ ] **Step 1: Create the docs page skeleton**

```markdown
# Large-dataset monotonic-depth screen

Screens deep vs shallow `absolute`-residual monotone stacks at each dataset's
full train size, and routes each dataset by the gate (Δ significant *and* ≥ the
practical margin) to a full [size-ladder](loan-size-ladder.md) study or the
standard benchmark. Method: {doc}`protocol` and the
[design spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-07-11-large-dataset-screen-design.md).

```{note}
Populated (table + plot) by the GPU session per `benchmarks/RUNBOOK-large-screen.md`.
```
```

- [ ] **Step 2: Create the RUNBOOK**

Write `benchmarks/RUNBOOK-large-screen.md` mirroring `RUNBOOK-loan-ladder.md`: (1) `git lfs pull`; (2) per-dataset `python -m benchmarks.large_screen_run --dataset <name> --out benchmarks/results/screen/<name>.json` (GPU-pin via `$MONONET_TORCH_DEVICE`); (3) render via `screen_report.render_screen_plot(records, Path('docs/_static/large-dataset-screen.png'))` + `screen_table`; (4) paste table into the docs page, commit the plot (PNG+PDF) and result JSONs; (5) `sphinx-build -W`.

- [ ] **Step 3: Wire the toctree**

Add `large-dataset-screen` to the benchmarks toctree in `docs/benchmarks/index.md` (follow the existing entry style).

- [ ] **Step 4: Verify the docs build**

Run: `LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html`
Expected: `build succeeded`, no warnings.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/large-dataset-screen.md docs/benchmarks/index.md benchmarks/RUNBOOK-large-screen.md
git commit -m "docs: large-dataset screen page skeleton + RUNBOOK + toctree"
```

---

## Out of scope for this plan (follow-on)

The remaining roster datasets (Lending Club-Zenodo, Home Credit [script-only],
Give Me Some Credit, Taiwan Credit, Polish Bankruptcy, German Credit) are each a
**mechanical repeat of Task 7**: add a `DataSource` to `SOURCES`, a `DatasetSpec`
with domain monotone directions (per the spec's roster table), a
`prepare/<name>.py` with a synthetic-sample test, an `_BUDGET` entry, and (for
redistributable ones) committed LFS gz data. Each needs its **raw file schema in
hand**, so they are planned in a follow-on once the raw sources are downloaded —
one task per dataset, same six steps. The infrastructure (Tasks 1–6, 8) and the
Adult reference (Task 7) make the path turnkey.

## Self-review notes

- **Spec coverage:** §4.1 data plumbing → Tasks 1–3, 7; §4.2 devcontainers →
  Task 1; §4.3 screen → Task 5; §4.4 gate → Task 4; §4.6 report/docs → Tasks 6,
  8. §3 roster → Adult in Task 7 + the follow-on note for the rest. §5 testing →
  each task's TDD steps. **Gap acknowledged & scoped:** the non-Adult roster is
  the documented follow-on above (raw schemas required).
- **Types:** `screen_dataset` returns exactly the keys `screen_table`/
  `render_screen_plot`/`gate` consume; `require_dataset` feeds `load(..., data_dir=)`;
  `prepare_adult`'s `MONO_INCREASING` matches the `DatasetSpec` monotone names.
