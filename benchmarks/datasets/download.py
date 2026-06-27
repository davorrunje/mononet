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
    """Get the default dataset cache directory.

    Respects `MONONET_DATA_DIR` environment variable if set.

    :returns: Path to dataset cache directory.
    """
    env = os.environ.get("MONONET_DATA_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "mononet" / "datasets"


def sha256(path: Path) -> str:
    """Compute SHA-256 digest of a file.

    :param path: Path to file.
    :returns: Hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, expected: str) -> bool:
    """Verify file checksum against expected SHA-256 digest.

    :param path: Path to file.
    :param expected: Expected hexadecimal SHA-256 digest.
    :returns: `True` if file exists and checksum matches, `False` otherwise.
    """
    return path.exists() and sha256(path) == expected


def _checksums() -> dict[str, str]:
    if not _MANIFEST.exists():
        return {}
    data = tomllib.loads(_MANIFEST.read_text(encoding="utf-8"))
    sums = data.get("sha256", {})
    return sums if isinstance(sums, dict) else {}


def download_all(dest: Path | None = None, *, force: bool = False) -> list[Path]:
    """Download all paper datasets from Zenodo.

    Skips files that already exist and pass verification (unless `force=True`).
    Missing checksums in manifest skip verification.

    :param dest: Destination directory; defaults to `default_dest()`.
    :param force: Re-download even if verified copy exists.
    :returns: List of downloaded file paths.
    :raises RuntimeError: If checksum verification fails after download.
    """
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
        urllib.request.urlretrieve(url, target)
        if want and not verify(target, want):
            raise RuntimeError(f"checksum mismatch for {name} after download")
        out.append(target)
    return out


def main() -> None:
    """CLI entry point for downloading datasets."""
    ap = argparse.ArgumentParser(description="Download mononet paper datasets")
    ap.add_argument("--dest", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    paths = download_all(args.dest, force=args.force)
    print(f"Downloaded {len(paths)} files to {paths[0].parent}")  # noqa: T201


if __name__ == "__main__":
    main()
