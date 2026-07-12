# SPDX-License-Identifier: Apache-2.0
"""JSON read/write for application result artifacts.

Result artifacts are small JSON mappings written to ``results/`` and consumed
by report/notebook code. Mirrors the conventions used in ``benchmarks/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_result(path: str | Path, obj: dict[str, Any]) -> None:
    """Write `obj` as sorted, pretty-printed JSON to `path`.

    Parent directories are created if missing.

    :param path: Destination file path.
    :param obj: JSON-serializable result mapping.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def read_result(path: str | Path) -> dict[str, Any]:
    """Read a JSON result mapping from `path`.

    :param path: Source file path.
    :returns: The parsed mapping.
    """
    data: dict[str, Any] = json.loads(Path(path).read_text())
    return data
