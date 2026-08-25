# SPDX-License-Identifier: Apache-2.0
"""Tests for the NGSIM download helper (no network: selection logic + URL only)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from applications.pinn.data import download_ngsim

if TYPE_CHECKING:
    from pathlib import Path


def test_i80_zip_url_is_deterministic() -> None:
    """The I-80 download URL embeds the dataset id, asset id, and filename."""
    url = download_ngsim.i80_zip_url()
    assert "8ect-6jqj" in url
    assert "ea269540-b86c-4b2d-a9c2-c8f4c0a3d0a0" in url
    assert "I-80-Emeryville-CA.zip" in url


def _touch(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def test_select_trajectory_csv_prefers_csv_and_period(tmp_path: Path) -> None:
    """Selection picks the period's .csv over the larger .txt and other periods."""
    base = tmp_path / "extracted" / "vehicle-trajectory-data"
    _touch(base / "0400pm-0415pm" / "trajectories-0400-0415.csv", 100)
    _touch(base / "0500pm-0515pm" / "trajectories-0500-0515.csv", 200)
    # A larger .txt sibling that must NOT be chosen (header-less, space-delimited).
    _touch(base / "0500pm-0515pm" / "trajectories-0500-0515.txt", 999)
    _touch(base / "detector-data" / "detector-data.csv", 50)

    chosen = download_ngsim._select_trajectory_csv(tmp_path, "0500-0515")
    assert chosen.name == "trajectories-0500-0515.csv"
    assert chosen.suffix == ".csv"


def test_select_trajectory_csv_falls_back_to_largest(tmp_path: Path) -> None:
    """With no period match, fall back to the largest trajectory-like csv."""
    base = tmp_path / "vtd"
    _touch(base / "trajectories-all.csv", 300)
    _touch(base / "notes.csv", 10)
    chosen = download_ngsim._select_trajectory_csv(tmp_path, "9999-9999")
    assert chosen.name == "trajectories-all.csv"
