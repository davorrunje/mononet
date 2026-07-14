# SPDX-License-Identifier: Apache-2.0
r"""Download the raw NGSIM I-80 trajectories and stage them as ``raw/i80.csv``.

Fetches the **I-80 Emeryville** vehicle-trajectory zip attachment from the
official ITS DataHub dataset (*Next Generation Simulation (NGSIM) Vehicle
Trajectories and Supporting Data*, U.S. DOT, public domain), extracts the chosen
15-minute period's trajectory CSV, and writes it to the path the preprocessor
(:mod:`applications.pinn.data.ngsim`) reads.

Stdlib only (``urllib`` + ``zipfile``) so it adds no dependencies.

**Network:** the dev container's egress to ``data.transportation.gov`` is
firewalled, so this must be run on a machine/CI with internet. The download URL
is deterministic (dataset ``8ect-6jqj``, I-80 asset
``ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0``); override it with ``--url`` if the
Socrata blobstore path changes.

Example::

    uv run python -m applications.pinn.data.download_ngsim
    # -> applications/pinn/data/raw/i80.csv  (period 0500-0515 by default)
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

_DATASET_ID = "8ect-6jqj"
_I80_ASSET_ID = "ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0"
_I80_FILENAME = "I-80-Emeryville-CA.zip"
_DEFAULT_PERIOD = "0500-0515"
_DEFAULT_OUT = "applications/pinn/data/raw/i80.csv"


def i80_zip_url() -> str:
    """Return the deterministic ITS DataHub download URL for the I-80 zip.

    :returns: The Socrata blobstore URL for ``I-80-Emeryville-CA.zip``.
    """
    return (
        f"https://data.transportation.gov/api/views/{_DATASET_ID}"
        f"/files/{_I80_ASSET_ID}?download=true&filename={_I80_FILENAME}"
    )


def _extract_all(zip_path: Path, dest: Path, *, depth: int = 2) -> None:
    """Extract ``zip_path`` into ``dest``, recursively unpacking nested zips.

    NGSIM site packages sometimes nest per-period data in inner zips.

    :param zip_path: Zip file to extract.
    :param dest: Destination directory.
    :param depth: Remaining nested-zip levels to unpack.
    """
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    if depth <= 0:
        return
    for inner in list(dest.rglob("*.zip")):
        sub = inner.with_suffix("")
        sub.mkdir(exist_ok=True)
        _extract_all(inner, sub, depth=depth - 1)


def _select_trajectory_csv(root: Path, period: str) -> Path:
    """Find the trajectory CSV for ``period`` under an extracted NGSIM tree.

    Prefers a file whose path contains every digit-group of ``period`` (e.g.
    ``0500`` and ``0515``) and looks like trajectory data; falls back to the
    largest ``.csv``/``.txt`` in the tree.

    :param root: Directory the zip was extracted into.
    :param period: Period token such as ``"0500-0515"``.
    :returns: Path to the selected trajectory file.
    :raises FileNotFoundError: If no ``.csv``/``.txt`` file is present.
    """
    tokens = [t for t in period.replace("_", "-").split("-") if t]
    candidates = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".csv", ".txt"}
    ]
    if not candidates:
        raise FileNotFoundError(f"no .csv/.txt trajectory file under {root}")

    def _looks_traj(p: Path) -> bool:
        return "trajector" in p.name.lower()

    # Prefer a parseable comma-delimited .csv (with a header) over the
    # whitespace-delimited, header-less .txt sibling, then the largest.
    def _rank(p: Path) -> tuple[bool, int]:
        return (p.suffix.lower() == ".csv", p.stat().st_size)

    period_hits = [
        p for p in candidates if all(tok in str(p).lower() for tok in tokens)
    ]
    traj_period = [p for p in period_hits if _looks_traj(p)]
    for pool in (traj_period, period_hits, [p for p in candidates if _looks_traj(p)]):
        if pool:
            return max(pool, key=_rank)
    return max(candidates, key=_rank)


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (with a browser-ish User-Agent)."""
    req = urllib.request.Request(url, headers={"User-Agent": "mononet-ngsim/1.0"})
    with urllib.request.urlopen(req) as resp, dest.open("wb") as fh:
        shutil.copyfileobj(resp, fh)


def build_raw_csv(
    out_csv: Path,
    *,
    period: str = _DEFAULT_PERIOD,
    cache_dir: Path | None = None,
    url: str | None = None,
) -> Path:
    """Download + extract the I-80 zip and stage the period's trajectory CSV.

    :param out_csv: Destination CSV path (e.g. ``.../raw/i80.csv``).
    :param period: 15-minute period token (default ``"0500-0515"``).
    :param cache_dir: Where to keep the downloaded zip / extraction (defaults to
        ``<out_csv parent>/.ngsim-cache``); the zip is reused if already present.
    :param url: Override download URL (defaults to :func:`i80_zip_url`).
    :returns: ``out_csv``.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else out_csv.parent / ".ngsim-cache"
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / _I80_FILENAME
    if not zip_path.exists():
        print(f"downloading {_I80_FILENAME} ...", flush=True)  # noqa: T201
        _download(url or i80_zip_url(), zip_path)
    extract_dir = cache / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir()
    _extract_all(zip_path, extract_dir)
    src = _select_trajectory_csv(extract_dir, period)
    shutil.copyfile(src, out_csv)
    print(f"selected {src.relative_to(extract_dir)} -> {out_csv}", flush=True)  # noqa: T201
    return out_csv


def main() -> None:
    """CLI: download the I-80 trajectories and write ``raw/i80.csv``."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=_DEFAULT_OUT)
    p.add_argument(
        "--period",
        default=_DEFAULT_PERIOD,
        help="15-min period token; 0500-0515 (congested) has the richest waves.",
    )
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--url", default=None, help="Override the download URL.")
    args = p.parse_args()
    out = build_raw_csv(
        Path(args.out),
        period=args.period,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        url=args.url,
    )
    print(f"== wrote {out} ==", flush=True)  # noqa: T201
    print("next: uv run python -m applications.pinn.data.ngsim --raw", out, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
