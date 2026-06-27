"""Generic CSV -> DatasetBundle loader."""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

import numpy as np

from benchmarks._common.bundle import DatasetBundle

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.datasets.spec import DatasetSpec


def _read_csv(path: Path) -> tuple[list[str], np.ndarray]:
    """Read a CSV file and return the header and data as a float64 array.

    :param path: Path to the CSV file.
    :returns: Tuple of (header_names, data_array).
    :raises ValueError: If the file is empty or has no data rows.
    """
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [[float(v) for v in r] for r in reader if r]
    return header, np.array(rows, dtype=np.float64)


def load_spec(spec: DatasetSpec, *, data_dir: Path) -> DatasetBundle:
    """Load a dataset from CSV files in *data_dir* according to *spec*.

    Expects `train_<name>.csv` and `test_<name>.csv` in *data_dir*.
    The CSV must have a header row; column names are resolved against
    `spec.mono_increasing` and `spec.mono_decreasing` — a typo in a
    column name raises :exc:`ValueError` at load time.

    :param spec: Dataset descriptor with column layout.
    :param data_dir: Directory containing the train/test CSV files.
    :returns: Populated :class:`~benchmarks._common.bundle.DatasetBundle`.
    :raises KeyError: If *spec.name* is not found in the registry.
    :raises ValueError: If a declared monotone column name is not in the header.
    """
    header, train = _read_csv(data_dir / f"train_{spec.name}.csv")
    _, test = _read_csv(data_dir / f"test_{spec.name}.csv")

    try:
        tgt = header.index(spec.target)
    except ValueError as exc:
        raise ValueError(
            f"Target column {spec.target!r} not found in header: {header}"
        ) from exc

    feat_idx = [i for i in range(len(header)) if i != tgt]
    names = tuple(header[i] for i in feat_idx)
    name_to_col: dict[str, int] = {n: c for c, n in enumerate(names)}

    missing_inc = [n for n in spec.mono_increasing if n not in name_to_col]
    missing_dec = [n for n in spec.mono_decreasing if n not in name_to_col]
    if missing_inc or missing_dec:
        raise ValueError(
            f"Monotone column names not found in header — "
            f"increasing: {missing_inc}, decreasing: {missing_dec}"
        )

    inc = tuple(name_to_col[n] for n in spec.mono_increasing)
    dec = tuple(name_to_col[n] for n in spec.mono_decreasing)

    return DatasetBundle(
        name=spec.name,
        task=spec.task,
        X_train=train[:, feat_idx],
        y_train=train[:, tgt],
        X_test=test[:, feat_idx],
        y_test=test[:, tgt],
        mono_increasing=inc,
        mono_decreasing=dec,
        feature_names=names,
        metadata={"source": "zenodo:7968969", "license": "CC-BY-4.0"},
    )
