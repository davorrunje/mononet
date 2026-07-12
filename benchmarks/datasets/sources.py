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

    :param name: Identifier for the dataset (used as a key in SOURCES and in
        filenames).
    :param hosting: ``lfs`` (committed under ``benchmarks/data/<name>/``) or
        ``script`` (user regenerates into the local cache from restricted raw).
    :param license: License identifier or human-readable licensing info.
    :param url: URL to the dataset source or documentation.
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
    "taiwan": DataSource(
        name="taiwan",
        hosting="lfs",
        license="CC-BY-4.0 (UCI)",
        url="https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
        prep_hint="committed via LFS; regenerate with prepare/taiwan.py",
    ),
    "polish": DataSource(
        name="polish",
        hosting="lfs",
        license="CC-BY-4.0 (UCI)",
        url="https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip",
        prep_hint="committed via LFS; regenerate with prepare/polish.py",
    ),
    "german": DataSource(
        name="german",
        hosting="lfs",
        license="CC-BY-4.0 (UCI)",
        url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        prep_hint="committed via LFS; regenerate with prepare/german.py",
    ),
    "lc": DataSource(
        name="lc",
        hosting="lfs",
        license="CC-BY-4.0 (Zenodo)",
        url="https://zenodo.org/records/11295916",
        prep_hint="committed via LFS; regenerate with prepare/lc.py",
    ),
}


def resolve_dir(name: str) -> Path:
    """Directory that should contain ``train_<name>.csv[.gz]`` for *name*.

    :param name: Dataset identifier.
    :returns: Path to the directory expected to contain dataset CSV files.
    """
    src = SOURCES[name]
    return _DATA_ROOT / name if src.hosting == "lfs" else default_dest()


def require_dataset(name: str) -> Path:
    """Resolve *name*'s data dir, asserting the train file is present.

    :param name: Dataset identifier.
    :returns: Path to the dataset directory if the train file exists.
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
