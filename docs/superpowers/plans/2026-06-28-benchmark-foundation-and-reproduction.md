# Benchmark Foundation + Paper Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the repo-only `benchmarks/` harness — Zenodo data download, dataset loaders, an embedding-composition model builder over the four mononet flavors × three backends, a runner, baselines, CI smoke tests, and reproduction notebooks — and validate it against the paper's numbers.

**Architecture:** A new top-level `benchmarks/` package (NOT shipped in the wheel). Pure value types and aggregation are unit-tested directly; the model builder and runner are tested on tiny synthetic bundles and ~50-row committed CSV fixtures (no network in CI). The headline reproduction numbers come from a manual maintainer run, not CI.

**Tech Stack:** Python 3.11+, numpy, stdlib `urllib` (downloader), the Sub-project A `mononet.{torch,jax,keras}` layers, xgboost, pytest, Sphinx/myst-nb.

## Global Constraints

- `benchmarks/` is repo-only: it MUST NOT be added to `[tool.hatch.build.targets.{sdist,wheel}]` (those include `["mononet"]` only). No `mononet` package change.
- Data is fetched to `~/.cache/mononet/datasets/` (override `--dest` / env `MONONET_DATA_DIR`), verified against a committed SHA-256 manifest, and **never** committed to git or LFS. The 10 files: `{train,test}_{auto,blog,compas,heart,loan}.csv` from `https://zenodo.org/records/7968969/files/<name>.csv?download=1` (CC-BY-4.0, DOI 10.5281/zenodo.7968969).
- The four flavors are `mode ∈ {"switch","absolute"}` × `residual ∈ {False, True}`, applied to the **monotone stack** only.
- Partial monotonicity uses the **embedding-composition**: non-monotone columns → unconstrained MLP → embedding; concat with sign-flipped monotone columns; feed a monotone (all non-decreasing) stack; linear head (regression) or sigmoid head (binary).
- `MonotonicityMask` is `{-1,+1}` only (Sub-project A). The monotone branch sign vector is `+1` for increasing, `-1` for decreasing monotone columns.
- Headline metric statistic: **mean ± std of the best 5 of 10 seeds**; also record plain 10-seed mean/std.
- Per-dataset HPs are transcribed from `airtai/monotonic-nn nbs/experiments/*.ipynb` (values in Task 8). Activation is `elu` everywhere.
- CI runs smoke tests only — no Zenodo download, no notebook execution, no GPU. Notebooks are committed with outputs from a manual run.
- Backends are optional extras; benchmark tests must `pytest.importorskip` the active backend and key off `MONONET_TEST_BACKEND` like the existing suite.
- Branch: `feat/benchmark-foundation` (already created, holds the spec). Never commit to `main`. Commits signed (Secretive SSH); end messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer. All commands run from repo root `/Users/davor/Projects/PhD/mononet`.

---

### Task 1: Package scaffold, `pyproject` deps, and `DatasetBundle`

**Files:**
- Create: `benchmarks/__init__.py`, `benchmarks/_common/__init__.py`, `benchmarks/_common/bundle.py`, `benchmarks/README.md`
- Modify: `pyproject.toml` (`bench` group; ruff includes; mypy/pytest paths)
- Test: `tests/benchmarks/__init__.py`, `tests/benchmarks/test_bundle.py`

**Interfaces:**
- Produces: `DatasetBundle` (frozen dataclass) and `mono_signs(bundle) -> np.ndarray` (int8 `{-1,+1}` vector over the monotone columns, increasing first then decreasing) and `mono_columns(bundle) -> tuple[int,...]` / `free_columns(bundle) -> tuple[int,...]`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_bundle.py`:

```python
import numpy as np
from benchmarks._common.bundle import DatasetBundle, mono_signs, mono_columns, free_columns


def _bundle() -> DatasetBundle:
    X = np.arange(12, dtype=np.float64).reshape(3, 4)
    y = np.array([0.0, 1.0, 0.0])
    return DatasetBundle(
        name="t", task="binary_classification",
        X_train=X, y_train=y, X_test=X, y_test=y,
        mono_increasing=(0,), mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"), metadata={},
    )


def test_mono_columns_increasing_then_decreasing():
    b = _bundle()
    assert mono_columns(b) == (0, 2)
    assert free_columns(b) == (1, 3)


def test_mono_signs_match_direction():
    b = _bundle()
    assert mono_signs(b).tolist() == [1, -1]
    assert mono_signs(b).dtype == np.int8


def test_bundle_is_frozen():
    b = _bundle()
    try:
        b.name = "x"  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("DatasetBundle must be frozen")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_bundle.py -q`
Expected: FAIL (module `benchmarks._common.bundle` not found).

- [ ] **Step 3: Implement** — `benchmarks/_common/bundle.py`:

```python
"""Canonical dataset container shared by all benchmark loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """A preprocessed dataset with declared monotonicity.

    :param mono_increasing: column indices the target is non-decreasing in.
    :param mono_decreasing: column indices the target is non-increasing in.
    """

    name: str
    task: Literal["binary_classification", "regression"]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    mono_increasing: tuple[int, ...]
    mono_decreasing: tuple[int, ...]
    feature_names: tuple[str, ...]
    metadata: dict[str, str]


def mono_columns(bundle: DatasetBundle) -> tuple[int, ...]:
    """Monotone column indices, increasing first then decreasing."""
    return (*bundle.mono_increasing, *bundle.mono_decreasing)


def free_columns(bundle: DatasetBundle) -> tuple[int, ...]:
    """Non-monotone column indices, in original order."""
    mono = set(mono_columns(bundle))
    return tuple(i for i in range(len(bundle.feature_names)) if i not in mono)


def mono_signs(bundle: DatasetBundle) -> np.ndarray:
    """`{-1,+1}` int8 sign per monotone column (`+1` increasing, `-1` decreasing)."""
    signs = [1] * len(bundle.mono_increasing) + [-1] * len(bundle.mono_decreasing)
    return np.array(signs, dtype=np.int8)
```

`benchmarks/__init__.py` and `benchmarks/_common/__init__.py` are empty docstring modules (`"""Benchmark harness (repo-only, not shipped)."""`).

- [ ] **Step 4: Modify `pyproject.toml`**

In `[project.optional-dependencies]` leave backends as-is. In `[dependency-groups].bench` add xgboost (keep `==` pin policy — pin to the latest resolved after `uv lock`):

```toml
bench = [
    "scikit-learn==1.8.0",
    "pandas==3.0.3",
    "matplotlib==3.10.9",
    "xgboost==3.0.6",
]
```

In `[tool.ruff].include` add `"benchmarks/**/*.py"`. In `[tool.mypy].files` add `"benchmarks"`. In `[tool.pytest.ini_options].testpaths` add `"tests/benchmarks"`. The downloader uses stdlib `urllib` — no runtime dep added.

Run `uv lock` and confirm only `xgboost` (and its deps) are added; if `3.0.6` is not the resolved version, use whatever `uv lock` resolves and record it. Then `uv sync --group bench`.

- [ ] **Step 5: Run the test to confirm it passes**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_bundle.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/ tests/benchmarks/ pyproject.toml uv.lock
git commit -m "$(cat <<'EOF'
bench: scaffold benchmarks package + DatasetBundle

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Config, result, seeds, and metric aggregation

**Files:**
- Create: `benchmarks/_common/config.py`, `benchmarks/_common/results.py`, `benchmarks/_common/seeds.py`
- Test: `tests/benchmarks/test_results.py`, `tests/benchmarks/test_config.py`

**Interfaces:**
- Produces: `BenchmarkConfig` (frozen, with `.replace(**kw)`), `OptimizerSpec`, `EarlyStoppingSpec`, `ResultRow`, `aggregate(rows, *, metric, best_of, top_k) -> Aggregate`, `seed_everything(backend, seed)`.

- [ ] **Step 1: Write the failing tests** — `tests/benchmarks/test_results.py`:

```python
import numpy as np
from benchmarks._common.results import ResultRow, aggregate


def _rows(values):
    return [
        ResultRow(dataset="auto", backend="torch", mode="switch", residual=False,
                  seed=i, scores={"mse": v}, epochs_run=50)
        for i, v in enumerate(values)
    ]


def test_best_5_of_10_takes_lowest_for_loss():
    # 10 seeds; "mse" is lower-is-better, so best 5 = the 5 smallest.
    rows = _rows([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    agg = aggregate(rows, metric="mse", lower_is_better=True, top_k=5)
    assert agg.n_seeds == 10 and agg.n_selected == 5
    assert np.isclose(agg.mean, np.mean([1, 2, 3, 4, 5]))
    assert np.isclose(agg.std, np.std([1, 2, 3, 4, 5]))


def test_best_5_of_10_takes_highest_for_accuracy():
    rows = _rows([0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
    agg = aggregate(rows, metric="mse", lower_is_better=False, top_k=5)
    assert np.isclose(agg.mean, np.mean([0.55, 0.56, 0.57, 0.58, 0.59]))
```

`tests/benchmarks/test_config.py`:

```python
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec


def _cfg() -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset="auto", backend="torch", mode="switch", residual=False,
        depth=2, width=21, activation="elu", convex_fraction=0.5,
        embed_hidden=(21,), dropout=0.0,
        optimizer=OptimizerSpec(name="adam", lr=1e-3, weight_decay=0.0),
        lr_decay=None, batch_size=16, epochs=50, early_stopping=None,
        seeds=(0, 1), metrics=("mse",),
    )


def test_replace_returns_modified_copy():
    c = _cfg()
    d = c.replace(mode="absolute", residual=True)
    assert c.mode == "switch" and c.residual is False
    assert d.mode == "absolute" and d.residual is True
    assert d.dataset == "auto"
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_results.py tests/benchmarks/test_config.py -q`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement** — `benchmarks/_common/config.py`:

```python
"""Benchmark run configuration (sweep-ready value objects)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class OptimizerSpec:
    name: Literal["adam"]
    lr: float
    weight_decay: float = 0.0


@dataclass(frozen=True, slots=True)
class EarlyStoppingSpec:
    monitor: str
    patience: int


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    dataset: str
    backend: Literal["torch", "jax", "keras"]
    mode: Literal["switch", "absolute"]
    residual: bool
    depth: int
    width: int
    activation: str
    convex_fraction: float
    embed_hidden: tuple[int, ...]
    dropout: float
    optimizer: OptimizerSpec
    lr_decay: float | None
    batch_size: int
    epochs: int
    early_stopping: EarlyStoppingSpec | None
    seeds: tuple[int, ...]
    metrics: tuple[Literal["accuracy", "rmse", "mse"], ...]

    def replace(self, **changes: Any) -> "BenchmarkConfig":
        """Return a copy with the given fields overridden."""
        return dataclasses.replace(self, **changes)
```

`benchmarks/_common/results.py`:

```python
"""Per-run result records and best-k-of-n aggregation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class ResultRow:
    dataset: str
    backend: str
    mode: str
    residual: bool
    seed: int
    scores: dict[str, float]
    epochs_run: int


@dataclass(frozen=True, slots=True)
class Aggregate:
    metric: str
    mean: float
    std: float
    n_seeds: int
    n_selected: int


def aggregate(
    rows: list[ResultRow], *, metric: str, lower_is_better: bool, top_k: int = 5
) -> Aggregate:
    """Mean/std of the best `top_k` rows by `metric`."""
    vals = np.array([r.scores[metric] for r in rows], dtype=np.float64)
    order = np.argsort(vals)
    selected = order[:top_k] if lower_is_better else order[::-1][:top_k]
    best = vals[selected]
    return Aggregate(
        metric=metric, mean=float(best.mean()), std=float(best.std()),
        n_seeds=len(rows), n_selected=int(best.size),
    )


def write_jsonl(rows: list[ResultRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
```

`benchmarks/_common/seeds.py`:

```python
"""Deterministic per-backend seeding."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(backend: str, seed: int) -> None:
    """Seed Python, NumPy, and the active backend's RNG."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if backend == "torch":
        import torch

        torch.manual_seed(seed)
    elif backend == "keras":
        import keras

        keras.utils.set_random_seed(seed)
    # jax is functional: callers thread an explicit key from `seed`.
```

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_results.py tests/benchmarks/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/ tests/benchmarks/test_results.py tests/benchmarks/test_config.py
git commit -m "$(cat <<'EOF'
bench: config, result, seeds, best-k-of-n aggregation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Zenodo download script + checksum manifest

**Files:**
- Create: `benchmarks/datasets/__init__.py`, `benchmarks/datasets/download.py`, `benchmarks/datasets/manifest.toml`
- Test: `tests/benchmarks/test_download.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `sha256(path) -> str`, `verify(path, expected) -> bool`, `default_dest() -> Path`, `download_all(dest, *, force=False) -> list[Path]`, and CLI `python -m benchmarks.datasets.download [--dest DIR] [--force]`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_download.py` (no network — tests checksum logic and skip behavior against a temp file):

```python
import hashlib
from pathlib import Path

from benchmarks.datasets.download import sha256, verify, FILES, ZENODO_URL


def test_sha256_matches_hashlib(tmp_path: Path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"hello,world\n1,2\n")
    assert sha256(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_verify_true_on_match_false_on_mismatch(tmp_path: Path):
    p = tmp_path / "f.csv"
    p.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert verify(p, digest) is True
    assert verify(p, "deadbeef") is False


def test_file_list_and_url_shape():
    assert set(FILES) == {
        f"{split}_{name}.csv"
        for split in ("train", "test")
        for name in ("auto", "blog", "compas", "heart", "loan")
    }
    assert ZENODO_URL.format(name="train_auto.csv").endswith(
        "/records/7968969/files/train_auto.csv?download=1"
    )
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_download.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — `benchmarks/datasets/download.py`:

```python
"""Download the paper's preprocessed datasets from Zenodo (CC-BY-4.0).

Source: https://zenodo.org/records/7968969 (DOI 10.5281/zenodo.7968969).
Files are written to a local cache and never committed to git or LFS.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tomllib
import urllib.request
from pathlib import Path

ZENODO_URL = "https://zenodo.org/records/7968969/files/{name}?download=1"
FILES = tuple(
    f"{split}_{name}.csv"
    for split in ("train", "test")
    for name in ("auto", "blog", "compas", "heart", "loan")
)
_MANIFEST = Path(__file__).with_name("manifest.toml")


def default_dest() -> Path:
    env = os.environ.get("MONONET_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "mononet" / "datasets"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str) -> bool:
    return path.exists() and sha256(path) == expected


def _checksums() -> dict[str, str]:
    if not _MANIFEST.exists():
        return {}
    data = tomllib.loads(_MANIFEST.read_text(encoding="utf-8"))
    return data.get("sha256", {})


def download_all(dest: Path | None = None, *, force: bool = False) -> list[Path]:
    dest = dest or default_dest()
    dest.mkdir(parents=True, exist_ok=True)
    expected = _checksums()
    out: list[Path] = []
    for name in FILES:
        target = dest / name
        want = expected.get(name)
        if not force and want and verify(target, want):
            out.append(target)
            continue
        url = ZENODO_URL.format(name=name)
        urllib.request.urlretrieve(url, target)  # noqa: S310 (trusted Zenodo URL)
        if want and not verify(target, want):
            raise RuntimeError(f"checksum mismatch for {name} after download")
        out.append(target)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Download mononet paper datasets")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    paths = download_all(args.dest, force=args.force)
    print(f"Downloaded {len(paths)} files to {paths[0].parent}")  # noqa: T201


if __name__ == "__main__":
    main()
```

`benchmarks/datasets/manifest.toml` (header + a `[sha256]` table). The checksums are filled by the maintainer after the first real download (`for f in ~/.cache/mononet/datasets/*.csv; do shasum -a 256 "$f"; done`). Ship the file with the header and an empty/partial table; missing entries skip verification (download still works), so this is not a blocking placeholder:

```toml
# Preprocessed datasets for "Constrained Monotonic Neural Networks" (ICML 2023).
# Source: https://zenodo.org/records/7968969  DOI: 10.5281/zenodo.7968969
# License: CC-BY-4.0 (Runje & Shankaranarayana). NOT committed to git/LFS.
# Maintainer: fill in real SHA-256 sums after the first download.
[sha256]
```

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_download.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/datasets/ tests/benchmarks/test_download.py
git commit -m "$(cat <<'EOF'
bench: Zenodo dataset downloader with checksum manifest

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Dataset loaders, descriptors, registry, and fixtures

**Files:**
- Create: `benchmarks/datasets/spec.py` (per-dataset descriptors), `benchmarks/datasets/loader.py` (generic CSV→bundle), `benchmarks/datasets/registry.py`
- Create fixtures: `tests/benchmarks/fixtures/{train,test}_auto.csv` (~30 rows sliced from the real file by the maintainer; for the test, a tiny hand-written CSV with the right columns is acceptable)
- Test: `tests/benchmarks/test_loaders.py`

**Interfaces:**
- Consumes: `DatasetBundle` (Task 1).
- Produces: `DATASETS: dict[str, DatasetSpec]`, `load(name, *, data_dir) -> DatasetBundle`.

`DatasetSpec` carries the column layout. The monotone columns/directions are the transcribed values:

| name | task | target col | mono_increasing (names) | mono_decreasing (names) |
|---|---|---|---|---|
| auto | regression | `mpg` | — | weight, displacement, horsepower |
| heart | binary | `target` | trestbps, chol | — |
| compas | binary | `two_year_recid` | priors_count, juv_fel_count, juv_misd_count, juv_other_count | — |
| loan | binary | `loan_status` | feature_1, feature_4 | feature_0, feature_2, feature_3 |
| blog | regression | `target` | feature_50..53, feature_55..59 (↑) | — |

> Column **names** must be confirmed against the actual CSV headers on first download; the loader resolves names→indices from the header row, so a name typo fails loudly. Blog's exact monotone set and its HPs (Task 8) are a flagged maintainer item.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_loaders.py` (uses a tiny fixture, not the cache):

```python
from pathlib import Path

import numpy as np
from benchmarks.datasets.registry import load, DATASETS

FIXTURES = Path(__file__).parent / "fixtures"


def test_auto_loader_shapes_and_monotonicity():
    b = load("auto", data_dir=FIXTURES)
    assert b.task == "regression"
    assert b.X_train.shape[1] == len(b.feature_names)
    # weight/displacement/horsepower declared decreasing, none increasing
    assert b.mono_decreasing and not b.mono_increasing
    assert b.X_test.shape[1] == b.X_train.shape[1]


def test_registry_lists_five_datasets():
    assert set(DATASETS) == {"auto", "blog", "compas", "heart", "loan"}
```

(The maintainer creates `fixtures/{train,test}_auto.csv` as a ~30-row slice with the real header. If the real header is unknown at implementation time, write a minimal CSV with columns `mpg,cylinders,displacement,horsepower,weight,acceleration,model_year` and a few rows so the loader test exercises the name→index resolution.)

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_loaders.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — `benchmarks/datasets/spec.py`:

```python
"""Per-dataset column descriptors (transcribed from airtai/monotonic-nn)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    task: Literal["binary_classification", "regression"]
    target: str
    mono_increasing: tuple[str, ...]
    mono_decreasing: tuple[str, ...]


DATASETS_SPEC: dict[str, DatasetSpec] = {
    "auto": DatasetSpec(
        "auto", "regression", "mpg",
        (), ("weight", "displacement", "horsepower"),
    ),
    "heart": DatasetSpec(
        "heart", "binary_classification", "target",
        ("trestbps", "chol"), (),
    ),
    "compas": DatasetSpec(
        "compas", "binary_classification", "two_year_recid",
        ("priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"), (),
    ),
    "loan": DatasetSpec(
        "loan", "binary_classification", "loan_status",
        ("feature_1", "feature_4"), ("feature_0", "feature_2", "feature_3"),
    ),
    "blog": DatasetSpec(
        "blog", "regression", "target",
        tuple(f"feature_{i}" for i in (50, 51, 52, 53, 55, 56, 57, 58, 59)), (),
    ),
}
```

`benchmarks/datasets/loader.py` reads `train_<name>.csv`/`test_<name>.csv` from `data_dir` with `numpy`/`csv` (header row → name→index), separates the target column, resolves the monotone names to indices, and returns a `DatasetBundle`. `registry.py` exposes `DATASETS = DATASETS_SPEC` and `load(name, *, data_dir)`.

```python
# benchmarks/datasets/loader.py
"""Generic CSV -> DatasetBundle loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from benchmarks._common.bundle import DatasetBundle
from benchmarks.datasets.spec import DatasetSpec


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    return header, np.array(rows, dtype=np.float64)


def load_spec(spec: DatasetSpec, *, data_dir: Path) -> DatasetBundle:
    header, train = _read_csv(data_dir / f"train_{spec.name}.csv")
    _, test = _read_csv(data_dir / f"test_{spec.name}.csv")
    tgt = header.index(spec.target)
    feat_idx = [i for i in range(len(header)) if i != tgt]
    names = tuple(header[i] for i in feat_idx)
    name_to_col = {n: c for c, n in enumerate(names)}
    inc = tuple(name_to_col[n] for n in spec.mono_increasing)
    dec = tuple(name_to_col[n] for n in spec.mono_decreasing)
    return DatasetBundle(
        name=spec.name, task=spec.task,
        X_train=train[:, feat_idx], y_train=train[:, tgt],
        X_test=test[:, feat_idx], y_test=test[:, tgt],
        mono_increasing=inc, mono_decreasing=dec,
        feature_names=names,
        metadata={"source": "zenodo:7968969", "license": "CC-BY-4.0"},
    )
```

```python
# benchmarks/datasets/registry.py
"""Dataset name -> loader."""

from __future__ import annotations

from pathlib import Path

from benchmarks._common.bundle import DatasetBundle
from benchmarks.datasets.loader import load_spec
from benchmarks.datasets.spec import DATASETS_SPEC as DATASETS


def load(name: str, *, data_dir: Path) -> DatasetBundle:
    return load_spec(DATASETS[name], data_dir=data_dir)
```

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_loaders.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/datasets/ tests/benchmarks/test_loaders.py tests/benchmarks/fixtures/
git commit -m "$(cat <<'EOF'
bench: dataset specs, generic CSV loader, registry, fixtures

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Model builder — torch + monotonicity test utilities

**Files:**
- Create: `benchmarks/_common/model_builder.py` (torch path + shared dispatch)
- Test: `tests/benchmarks/test_model_builder_torch.py`

**Interfaces:**
- Consumes: `DatasetBundle`, `mono_columns`/`free_columns`/`mono_signs` (Task 1), `BenchmarkConfig` (Task 2), `mononet.torch` layers (Sub-project A).
- Produces: `build_model(cfg, bundle) -> object` returning a backend-native callable model; for torch an `nn.Module`. The monotone stack uses `mononet.torch.MonoLinear`/`MonoResidual` and `MonoInput`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_model_builder_torch.py`:

```python
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.model_builder import build_model


def _bundle(n=64, d=7):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = (X[:, 4] * -1.0 + rng.normal(scale=0.1, size=n)).astype(np.float64)
    return DatasetBundle(
        name="syn", task="regression",
        X_train=X, y_train=y, X_test=X, y_test=y,
        mono_increasing=(), mono_decreasing=(4,),
        feature_names=tuple(f"f{i}" for i in range(d)), metadata={},
    )


def _cfg(mode, residual):
    return BenchmarkConfig(
        dataset="syn", backend="torch", mode=mode, residual=residual,
        depth=2, width=8, activation="elu", convex_fraction=0.5,
        embed_hidden=(8,), dropout=0.0,
        optimizer=OptimizerSpec("adam", 1e-3, 0.0), lr_decay=None,
        batch_size=16, epochs=1, early_stopping=None, seeds=(0,), metrics=("mse",),
    )


@pytest.mark.parametrize("mode", ["switch", "absolute"])
@pytest.mark.parametrize("residual", [False, True])
def test_builds_and_output_shape(mode, residual):
    b = _bundle()
    model = build_model(_cfg(mode, residual), b)
    x = torch.tensor(b.X_train)
    out = model(x)
    assert out.shape == (b.X_train.shape[0], 1)


@pytest.mark.parametrize("mode", ["switch", "absolute"])
def test_monotone_in_decreasing_feature(mode):
    # Output must be non-increasing in column 4 (declared decreasing).
    b = _bundle()
    model = build_model(_cfg(mode, residual=False), b).eval()
    x = torch.tensor(b.X_train)
    x_hi = x.clone()
    x_hi[:, 4] += 1.0
    with torch.no_grad():
        assert torch.all(model(x_hi) <= model(x) + 1e-5)
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_model_builder_torch.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — `benchmarks/_common/model_builder.py` (torch path; dispatch stub for jax/keras filled in Task 6):

```python
"""Embedding-composition model builder over the four mononet flavors."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks._common.bundle import (
    DatasetBundle, free_columns, mono_columns, mono_signs,
)
from benchmarks._common.config import BenchmarkConfig


def build_model(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:
    if cfg.backend == "torch":
        return _build_torch(cfg, bundle)
    if cfg.backend == "jax":
        return _build_jax(cfg, bundle)
    if cfg.backend == "keras":
        return _build_keras(cfg, bundle)
    raise ValueError(cfg.backend)


def _build_torch(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:
    import torch
    from torch import nn

    from mononet.core.types import MonotonicityMask
    from mononet.torch import MonoInput, MonoLinear, MonoResidual

    mono_cols = list(mono_columns(bundle))
    free_cols = list(free_columns(bundle))
    signs = mono_signs(bundle)
    binary = bundle.task == "binary_classification"

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mono_cols = torch.tensor(mono_cols, dtype=torch.long)
            self.free_cols = torch.tensor(free_cols, dtype=torch.long)
            self.mono_input = (
                MonoInput(MonotonicityMask(signs)) if mono_cols else None
            )
            # unconstrained branch
            free_layers: list[nn.Module] = []
            in_f = len(free_cols)
            embed_out = 0
            if free_cols:
                for h in cfg.embed_hidden:
                    free_layers += [nn.Linear(in_f, h), nn.ELU()]
                    if cfg.dropout:
                        free_layers.append(nn.Dropout(cfg.dropout))
                    in_f = h
                embed_out = in_f
            self.free_mlp = nn.Sequential(*free_layers) if free_layers else None
            # monotone stack over concat([signed mono feats, embedding])
            stack_in = len(mono_cols) + embed_out
            mono_layers: list[nn.Module] = []
            prev = stack_in
            if cfg.residual:
                mono_layers.append(
                    MonoLinear(prev, cfg.width, mode=cfg.mode, activation=cfg.activation)
                )
                prev = cfg.width
                for _ in range(cfg.depth):
                    mono_layers.append(
                        MonoResidual(prev, cfg.width, mode=cfg.mode, activation=cfg.activation)
                    )
                    prev = cfg.width
            else:
                for _ in range(cfg.depth):
                    mono_layers.append(
                        MonoLinear(prev, cfg.width, mode=cfg.mode, activation=cfg.activation)
                    )
                    prev = cfg.width
            self.mono_stack = nn.Sequential(*mono_layers)
            self.head = MonoLinear(prev, 1, mode=cfg.mode)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            parts: list[torch.Tensor] = []
            if self.mono_input is not None:
                parts.append(self.mono_input(x.index_select(1, self.mono_cols.to(x.device))))
            if self.free_mlp is not None:
                parts.append(self.free_mlp(x.index_select(1, self.free_cols.to(x.device))))
            z = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
            y = self.head(self.mono_stack(z))
            return torch.sigmoid(y) if binary else y

    return Model().double()


def _build_jax(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:  # Task 6
    raise NotImplementedError("jax builder added in Task 6")


def _build_keras(cfg: BenchmarkConfig, bundle: DatasetBundle) -> Any:  # Task 6
    raise NotImplementedError("keras builder added in Task 6")
```

(Note: `MonoInput` applies the sign flips, so the monotone stack receives non-decreasing-oriented inputs; the head is a `MonoLinear` so the output stays monotone. `.double()` keeps the synthetic float64 test exact.)

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_model_builder_torch.py -q`
Expected: PASS (6 build cases + 2 monotonicity cases).

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/model_builder.py tests/benchmarks/test_model_builder_torch.py
git commit -m "$(cat <<'EOF'
bench: torch embedding-composition model builder (4 flavors)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Model builder — jax and keras paths

**Files:**
- Modify: `benchmarks/_common/model_builder.py` (`_build_jax`, `_build_keras`)
- Test: `tests/benchmarks/test_model_builder_jax.py`, `tests/benchmarks/test_model_builder_keras.py`

**Interfaces:**
- Consumes: `mononet.jax` (`MonoInput`, `MonoLinear`, `MonoResidual`, `nnx.Rngs`), `mononet.keras` (`MonoInput`, `MonoDense`, `MonoResidual`).
- Produces: `_build_jax`/`_build_keras` returning native callables with the same `(N,1)` output contract and the same embedding-composition topology as torch.

- [ ] **Step 1: Write the failing tests** — mirror `test_model_builder_torch.py` for each backend, using `pytest.importorskip("jax")` / `importorskip("keras")` (set `KERAS_BACKEND=jax`), the same synthetic bundle, asserting output shape `(N,1)` for all four flavors and non-increasing output in the declared-decreasing column for both modes. (Replicate the torch test's structure; do not import torch.)

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=jax uv run pytest tests/benchmarks/test_model_builder_jax.py -q`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement** — fill `_build_jax` and `_build_keras` mirroring the torch topology:
  - **jax (Flax NNX):** an `nnx.Module` taking `rngs=nnx.Rngs(seed)`; unconstrained branch = `nnx.Linear` + `jax.nn.elu` (+ `nnx.Dropout`); monotone stack = `mononet.jax.MonoLinear`/`MonoResidual`; concat via `jnp.concatenate`; sigmoid head for binary. Column selection via static index arrays.
  - **keras:** a `keras.Model` (functional or subclassed); unconstrained branch = `keras.layers.Dense(..., activation="elu")` (+ `Dropout`); monotone stack = `mononet.keras.MonoDense`/`MonoResidual`; `keras.layers.Concatenate`; final `MonoDense(1)` then `sigmoid` for binary. Keras infers input dims, so pass `units` only.

  Keep the column split, sign handling (`MonoInput`), and head identical to torch so the three builders are structurally the same model.

- [ ] **Step 4: Run to confirm pass**

Run:
```bash
MONONET_TEST_BACKEND=jax   uv run pytest tests/benchmarks/test_model_builder_jax.py -q
MONONET_TEST_BACKEND=keras KERAS_BACKEND=jax uv run pytest tests/benchmarks/test_model_builder_keras.py -q
```
Expected: PASS for both.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/model_builder.py tests/benchmarks/test_model_builder_jax.py tests/benchmarks/test_model_builder_keras.py
git commit -m "$(cat <<'EOF'
bench: jax and keras model builders (embedding-composition)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Runner + CLI

**Files:**
- Create: `benchmarks/_common/runner.py`, `benchmarks/run.py`
- Test: `tests/benchmarks/test_runner.py`

**Interfaces:**
- Consumes: `build_model` (Tasks 5-6), `BenchmarkConfig`/`ResultRow` (Task 2), `seed_everything` (Task 2), a `DatasetBundle`.
- Produces: `run(cfg, bundle) -> list[ResultRow]` (one row per seed) and CLI `python -m benchmarks.run --dataset auto --backend torch --mode switch [--residual] [--epochs N] [--seeds 0 1 ...] [--data-dir DIR]`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_runner.py`:

```python
import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.runner import run


def _bundle(n=128, d=6):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, d)).astype(np.float64)
    y = (X[:, 0] + 0.1 * rng.normal(size=n)).astype(np.float64)
    return DatasetBundle("syn", "regression", X, y, X, y, (0,), (),
                         tuple(f"f{i}" for i in range(d)), {})


def test_run_returns_one_row_per_seed_with_finite_metric():
    cfg = BenchmarkConfig(
        dataset="syn", backend="torch", mode="switch", residual=False,
        depth=1, width=8, activation="elu", convex_fraction=0.5,
        embed_hidden=(8,), dropout=0.0,
        optimizer=OptimizerSpec("adam", 1e-2, 0.0), lr_decay=None,
        batch_size=32, epochs=3, early_stopping=None, seeds=(0, 1), metrics=("mse",),
    )
    rows = run(cfg, _bundle())
    assert len(rows) == 2
    assert all(np.isfinite(r.scores["mse"]) for r in rows)
    assert all(r.seed in (0, 1) for r in rows)
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_runner.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement** — `benchmarks/_common/runner.py`. For each seed: `seed_everything`, build model, train for `cfg.epochs` (per-backend training loop: torch = Adam + MSELoss/BCELoss minibatches; jax = optax adam; keras = `model.compile(...).fit(...)`), evaluate on test set, compute the requested metrics (`mse`, `rmse=sqrt(mse)`, `accuracy` thresholded at 0.5), append a `ResultRow`. `weight_decay` → optimizer; `lr_decay` → per-epoch LR multiply; `dropout` already in the model. Keep the training loop minimal but real (no mocks). `run.py` parses args, loads the bundle via `registry.load(dataset, data_dir=...)` (defaulting to `download.default_dest()`), builds a `BenchmarkConfig` (HPs from the dataset's TOML in Task 8, overridable by flags), calls `run`, writes JSONL via `results.write_jsonl`, prints the aggregate.

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_runner.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/_common/runner.py benchmarks/run.py tests/benchmarks/test_runner.py
git commit -m "$(cat <<'EOF'
bench: training/eval runner and run.py CLI

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Per-dataset configs

**Files:**
- Create: `benchmarks/configs/{auto,heart,compas,loan,blog}.toml`, `benchmarks/_common/config_io.py`
- Test: `tests/benchmarks/test_config_io.py`

**Interfaces:**
- Produces: `load_config(path, *, backend, mode, residual) -> BenchmarkConfig`.

- [ ] **Step 1: Write the failing test** — `tests/benchmarks/test_config_io.py`:

```python
from pathlib import Path

from benchmarks._common.config_io import load_config

CONFIGS = Path("benchmarks/configs")


def test_auto_config_values():
    cfg = load_config(CONFIGS / "auto.toml", backend="torch", mode="switch", residual=False)
    assert cfg.dataset == "auto" and cfg.depth == 2 and cfg.width == 21
    assert cfg.activation == "elu" and cfg.batch_size == 16 and cfg.epochs == 50
    assert abs(cfg.optimizer.lr - 0.073407) < 1e-9
    assert cfg.backend == "torch" and cfg.mode == "switch" and cfg.residual is False
```

- [ ] **Step 2: Run to confirm failure**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_config_io.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement the configs** (transcribed values). `benchmarks/configs/auto.toml`:

```toml
# Source: airtai/monotonic-nn nbs/experiments/AutoMPG.ipynb
[dataset]
name = "auto"
[model]
depth = 2
width = 21
activation = "elu"
convex_fraction = 0.5
embed_hidden = [21]
[train]
optimizer = "adam"
lr = 0.073407
weight_decay = 0.058583
dropout = 0.157718
lr_decay = 0.887923
batch_size = 16
epochs = 50
seeds = [0,1,2,3,4,5,6,7,8,9]
metrics = ["mse"]
```

`heart.toml` (depth 2, width 22, lr 0.001, wd 0.113929, dropout 0.397874, lr_decay 0.894921, batch 16, epochs 50, metrics `["accuracy"]`, embed_hidden [22]).
`compas.toml` (depth 2, width 27, lr 0.084685, wd 0.137518, dropout 0.175917, lr_decay 0.899399, batch 8, epochs 50, metrics `["accuracy"]`, embed_hidden [27]).
`loan.toml` (depth 2, width 8, lr 0.008, wd 0.0, dropout 0.0, lr_decay 1.0, batch 256, epochs 20, metrics `["accuracy"]`, embed_hidden [8]).
`blog.toml` — header notes HPs are **not exposed** in `Blog.ipynb`'s rendered cells; ship regression defaults (depth 2, width 21, lr 0.001, wd 0.0, dropout 0.0, lr_decay 1.0, batch 128, epochs 50, metrics `["rmse"]`, embed_hidden [21]) with a comment: `# PROVISIONAL — maintainer: confirm against Blog.ipynb before the headline run.`

`benchmarks/_common/config_io.py` reads the TOML with `tomllib`, fills `BenchmarkConfig` (backend/mode/residual from the args, convex_fraction default 0.5, early_stopping None).

- [ ] **Step 4: Run to confirm pass**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_config_io.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/configs/ benchmarks/_common/config_io.py tests/benchmarks/test_config_io.py
git commit -m "$(cat <<'EOF'
bench: per-dataset configs from airtai notebooks + config IO

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: XGBoost baseline

**Files:**
- Create: `benchmarks/baselines/__init__.py`, `benchmarks/baselines/xgboost.py`
- Test: `tests/benchmarks/test_baseline_xgboost.py`

**Interfaces:**
- Consumes: `DatasetBundle`.
- Produces: `run_xgboost(bundle, *, seed=0) -> dict[str, float]` returning the dataset's metric(s).

- [ ] **Step 1: Write the failing test**:

```python
import numpy as np
from benchmarks._common.bundle import DatasetBundle
from benchmarks.baselines.xgboost import run_xgboost


def test_xgboost_regression_finite_mse():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)); y = X[:, 0] + 0.1 * rng.normal(size=200)
    b = DatasetBundle("syn", "regression", X, y, X, y, (0,), (),
                      tuple(f"f{i}" for i in range(5)), {})
    scores = run_xgboost(b, seed=0)
    assert np.isfinite(scores["mse"])
```

- [ ] **Step 2: Run to confirm failure** — `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks/test_baseline_xgboost.py -q` → FAIL.

- [ ] **Step 3: Implement** — `run_xgboost` picks `XGBRegressor` (regression → mse/rmse) or `XGBClassifier` (binary → accuracy), fits on train, scores on test.

- [ ] **Step 4: Run to confirm pass** — same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/baselines/ tests/benchmarks/test_baseline_xgboost.py
git commit -m "$(cat <<'EOF'
bench: XGBoost baseline

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 10: CI smoke test

**Files:**
- Create: `tests/benchmarks/test_smoke.py`
- Modify: `.github/workflows/build.yml` (extend the existing `test` job command to include `tests/benchmarks`)

**Interfaces:**
- Consumes: everything above. No network, no Zenodo cache.

- [ ] **Step 1: Write the smoke test** — for the active `MONONET_TEST_BACKEND`, build + train 2 epochs on a synthetic bundle for one flavor and assert a finite `ResultRow`; also load the `auto` fixture and assert shapes (reuses Task 4 fixtures):

```python
import os
import numpy as np
import pytest

BACKEND = os.environ.get("MONONET_TEST_BACKEND", "torch")
pytest.importorskip(BACKEND if BACKEND != "keras" else "keras")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec
from benchmarks._common.runner import run


def test_smoke_one_flavor_trains():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(96, 5)); y = (X[:, 0] > 0).astype(float)
    b = DatasetBundle("syn", "binary_classification", X, y, X, y, (0,), (),
                      tuple(f"f{i}" for i in range(5)), {})
    cfg = BenchmarkConfig(
        dataset="syn", backend=BACKEND, mode="switch", residual=False,
        depth=1, width=8, activation="elu", convex_fraction=0.5,
        embed_hidden=(8,), dropout=0.0, optimizer=OptimizerSpec("adam", 1e-2, 0.0),
        lr_decay=None, batch_size=32, epochs=2, early_stopping=None,
        seeds=(0,), metrics=("accuracy",),
    )
    rows = run(cfg, b)
    assert len(rows) == 1 and np.isfinite(rows[0].scores["accuracy"])
```

- [ ] **Step 2: Run per backend** — confirm PASS on torch, jax, keras (the existing matrix already sets `MONONET_TEST_BACKEND`/`KERAS_BACKEND`).

- [ ] **Step 3: Modify `build.yml`** — the `test` job's run line becomes:

```yaml
        run: pytest tests/core "tests/${{ matrix.backend }}" tests/equivalence tests/benchmarks tests/test_top_level_imports.py -v
```

- [ ] **Step 4: Confirm the full benchmark suite passes for one backend**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/benchmarks -q`
Expected: all PASS, output pristine.

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/test_smoke.py .github/workflows/build.yml
git commit -m "$(cat <<'EOF'
bench: CI smoke test across backends; wire into build matrix

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Reproduction notebooks + Sphinx docs

**Files:**
- Create: `docs/benchmarks/paper-reproduction/{index.md,auto-mpg.ipynb,heart-disease.ipynb,compas.ipynb,blog-feedback.ipynb,loan-defaulter.ipynb,tables.ipynb}`
- Modify: `docs/benchmarks/index.md` (or `docs/index.md` toctree) to link the new section
- Create: `benchmarks/results/.gitignore` (`*.jsonl`) so raw per-seed output is ignored; the aggregated `paper-reproduction.json` is committed by the maintainer after the headline run.

**Interfaces:** none (documentation deliverable).

- [ ] **Step 1: Create `index.md`** — one paragraph (what's reproduced, link to paper/Zenodo, how to fetch data: `python -m benchmarks.datasets.download`), and a toctree listing the five dataset notebooks + `tables`.

- [ ] **Step 2: Create the five dataset notebooks** — each is a thin scaffold: a markdown cell describing the dataset + its monotone features, then a code cell that loads the bundle and runs all four flavors via `benchmarks.run` (or imports `run` directly) for a small `--epochs`/seed budget, then a cell rendering the per-flavor metric vs the paper-quoted number. Notebooks are created **without executed outputs**; the maintainer executes them manually (`tools/execute-benchmarks.sh`) on the full data/seed budget and commits the outputs. Each notebook must be valid JSON and import-clean.

- [ ] **Step 3: Create `tables.ipynb`** — loads the committed `benchmarks/results/paper-reproduction.json` (or regenerates from a run), and renders the headline table (rows: paper-quoted, 4 flavors, XGBoost; per-dataset metric) plus the cross-backend agreement table.

- [ ] **Step 4: Wire the toctree and verify the docs build** (the new notebooks have no outputs yet, so the build must tolerate unexecuted notebooks — `myst-nb` `execute: false` is already set per Sub-project A docs config).

Run: `./tools/build-docs.sh`
Expected: exit 0; the "Reproducing the paper" section renders.

- [ ] **Step 5: Commit**

```bash
git add docs/benchmarks/paper-reproduction/ docs/benchmarks/index.md docs/index.md benchmarks/results/.gitignore
git commit -m "$(cat <<'EOF'
docs: paper-reproduction notebooks + Sphinx section

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Notes for the executor

- **No network in execution.** Tasks never download from Zenodo. Loader tests use tiny fixtures; the downloader is unit-tested by checksum logic only. The real data fetch, the SHA-256 manifest fill, the headline numbers, and the notebook outputs are **manual maintainer steps**, documented in `benchmarks/README.md`.
- **Backend gating.** Every backend-touching test uses `pytest.importorskip` and keys off `MONONET_TEST_BACKEND` exactly like the existing suite; CI runs one backend per matrix leg.
- **Blog is the one flagged gap:** its monotone columns are best-known (features 50–53, 55–59) and its HPs are provisional pending a maintainer read of `Blog.ipynb`. Everything else is concrete.
- **Reproduction fidelity is not asserted in tests.** Tests check the harness *works* (shapes, monotonicity, finite metrics). Matching the paper's numbers is verified by the maintainer's manual run against §6 of the spec (Auto MPG MSE ≈ 8.37 is the calibration anchor).
- `./tools/get-version.sh` uses PCRE grep and fails on macOS; irrelevant here.
- Do not commit the Zenodo CSVs, `dist/`, or raw `*.jsonl` results.
