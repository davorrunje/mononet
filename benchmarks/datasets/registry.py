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
    """Load a dataset by name.

    Generator-backed specs (``spec.generator is not None``, e.g. the
    ``synth_*`` depth-probe datasets) are produced on the fly by
    :func:`~benchmarks.datasets.synthetic.synth_monotone` and *data_dir* is
    ignored. Otherwise the dataset is loaded from ``train_<name>.csv`` and
    ``test_<name>.csv`` under *data_dir*.

    :param name: Dataset key, one of ``DATASETS``.
    :param data_dir: Directory containing the train/test CSV files (ignored
        for generator-backed specs).
    :returns: Populated :class:`~benchmarks._common.bundle.DatasetBundle`.
    :raises KeyError: If *name* is not in the registry.
    """
    spec = DATASETS[name]
    if spec.generator is not None:
        from benchmarks.datasets.synthetic import synth_monotone

        g = spec.generator
        return synth_monotone(
            g.kind, g.c, d=g.d, n_train=g.n_train, n_test=g.n_test, seed=g.seed
        )
    return load_spec(spec, data_dir=data_dir)
