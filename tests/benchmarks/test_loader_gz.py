from __future__ import annotations

import gzip
from typing import TYPE_CHECKING

from benchmarks.datasets.loader import load_spec
from benchmarks.datasets.spec import DatasetSpec

if TYPE_CHECKING:
    from pathlib import Path


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
    assert b.mono_increasing == (0,)
    assert b.mono_decreasing == (1,)
