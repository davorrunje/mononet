# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = ["packaging>=24"]
# ///
"""Expand `requires-python` into the concrete Python versions mononet supports.

`[project] requires-python` in `pyproject.toml` is the single source of truth for
which Python versions this package supports. This script expands that specifier
so the local type-check sweep (`tools/typecheck-all.sh`) and the CI matrix
consume one answer instead of two hand-maintained lists.

The PEP 723 header above lets it run without syncing the project environment:

    uv run --script tools/supported_pythons.py           # 3.11 3.12 3.13 3.14
    uv run --script tools/supported_pythons.py --json    # ["3.11", ...]

:param --json: Emit a JSON array instead of one version per line. The CI matrix
    consumes this via `fromJson`.
:param --requires-python: Use this specifier instead of reading `pyproject.toml`.
:param --pyproject: Path to the `pyproject.toml` to read.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.specifiers import SpecifierSet

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Candidate versions considered when expanding the specifier. The floor is
#: below anything this project ever supported; the ceiling only has to stay
#: ahead of CPython's release cadence.
CANDIDATES: tuple[str, ...] = tuple(f"3.{minor}" for minor in range(9, 21))

DEFAULT_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def expand(specifier: str, candidates: Sequence[str] = CANDIDATES) -> list[str]:
    """Return the candidates satisfying `specifier`, in ascending order.

    :param specifier: A PEP 440 version specifier, e.g. `">=3.11,<3.15"`.
    :param candidates: Versions to test. Defaults to `CANDIDATES`.
    :returns: The satisfying versions, e.g. `["3.11", "3.12", "3.13", "3.14"]`.
    """
    spec = SpecifierSet(specifier)
    return [version for version in candidates if spec.contains(version)]


def read_requires_python(pyproject: Path = DEFAULT_PYPROJECT) -> str:
    """Read `[project] requires-python` from a `pyproject.toml`.

    :param pyproject: Path to the file to read.
    :returns: The raw specifier string.
    :raises KeyError: If the `[project] requires-python` key is absent.
    :raises TypeError: If the key is present but not a string.
    """
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    requires = data["project"]["requires-python"]
    if not isinstance(requires, str):
        raise TypeError(
            f"{pyproject}: [project] requires-python must be a string, "
            f"got {type(requires).__name__}"
        )
    return requires


def main() -> None:
    """Print the supported versions, one per line or as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--requires-python", default=None)
    parser.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT)
    args = parser.parse_args()

    specifier = args.requires_python or read_requires_python(args.pyproject)
    versions = expand(specifier)
    if not versions:
        print(f"no candidate version satisfies {specifier!r}", file=sys.stderr)
        raise SystemExit(1)

    print(json.dumps(versions) if args.as_json else "\n".join(versions))


if __name__ == "__main__":
    main()
