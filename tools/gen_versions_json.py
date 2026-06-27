# SPDX-License-Identifier: Apache-2.0
"""Generate versions.json for the PyData Sphinx Theme version switcher.

Run after `sphinx-polyversion` has populated docs/_build/html/<version>/
directories. Writes versions.json at the root of the build output.

:param build_dir: Path to the multiversion build output
    (e.g. `docs/_build/html`).
:param base_url: The site's base URL (used to construct per-version URLs).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("base_url", type=str)
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    entries: list[dict[str, str | bool]] = []

    version_dirs = sorted(p.name for p in args.build_dir.iterdir() if p.is_dir())
    tagged = [v for v in version_dirs if VERSION_RE.match(v)]
    tagged.sort(reverse=True)  # newest first

    if "main" in version_dirs:
        entries.append(
            {
                "name": "dev (main)",
                "version": "latest",
                "url": f"{base}/main/",
            }
        )

    for v in tagged:
        entry: dict[str, str | bool] = {
            "name": v,
            "version": v,
            "url": f"{base}/{v}/",
        }
        if v == tagged[0]:
            entry["preferred"] = True
        entries.append(entry)

    out = args.build_dir / "versions.json"
    out.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
