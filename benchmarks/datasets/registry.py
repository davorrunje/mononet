"""Dataset name -> loader."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.datasets.loader import load_spec
from benchmarks.datasets.spec import DATASETS_SPEC as DATASETS

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks._common.bundle import DatasetBundle

__all__ = ["DATASETS", "load"]


def load(name: str, *, data_dir: Path) -> DatasetBundle:
    """Load a dataset by name from CSV files in *data_dir*.

    :param name: Dataset key, one of ``DATASETS``.
    :param data_dir: Directory containing ``train_<name>.csv`` and
        ``test_<name>.csv``.
    :returns: Populated :class:`~benchmarks._common.bundle.DatasetBundle`.
    :raises KeyError: If *name* is not in the registry.
    """
    return load_spec(DATASETS[name], data_dir=data_dir)
