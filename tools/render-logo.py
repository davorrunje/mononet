"""Rasterize the mononet logo SVGs to PNG / favicon outputs.

Reads ``docs/_static/logo-light.svg`` and writes:
- ``docs/_static/logo.png`` — 256x256 PNG (legacy / fallback consumers).
- ``docs/_static/favicon.png`` — 32x32 PNG (browser tab icon).

Idempotent: re-running with unchanged SVGs produces byte-identical PNGs.

This is a developer tool, not a CI step. Run it locally whenever
``logo-light.svg`` changes. Requires ``cairosvg`` (in the ``docs``
dependency group); cairosvg in turn needs the system library
``libcairo2`` (already present on the project's devcontainer image).
"""

from __future__ import annotations

from pathlib import Path

import cairosvg

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = REPO_ROOT / "docs" / "_static"

#: Source SVG used for both PNG renders.
SRC = STATIC_DIR / "logo-light.svg"

#: Output PNG renders. ``(destination, size_in_px)``.
TARGETS: tuple[tuple[Path, int], ...] = (
    (STATIC_DIR / "logo.png", 256),
    (STATIC_DIR / "favicon.png", 32),
)


def main() -> None:
    svg_bytes = SRC.read_bytes()
    for dst, size in TARGETS:
        cairosvg.svg2png(
            bytestring=svg_bytes,
            write_to=str(dst),
            output_width=size,
            output_height=size,
        )
        print(f"wrote {dst.relative_to(REPO_ROOT)} ({size}x{size})")


if __name__ == "__main__":
    main()
