from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarks.datasets.sources import SOURCES, DataSource, require_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch


def test_adult_is_lfs_hosted() -> None:
    src = SOURCES["adult"]
    assert isinstance(src, DataSource)
    assert src.hosting == "lfs"
    assert "uci" in src.url.lower() or "openml" in src.url.lower()


def test_taiwan_is_lfs_hosted() -> None:
    src = SOURCES["taiwan"]
    assert isinstance(src, DataSource)
    assert src.hosting == "lfs"
    assert "uci" in src.url.lower()


def test_polish_is_lfs_hosted() -> None:
    src = SOURCES["polish"]
    assert isinstance(src, DataSource)
    assert src.hosting == "lfs"
    assert "uci" in src.url.lower()


def test_require_dataset_missing_script_source_raises_actionable(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A script-only dataset with no local file raises with prep instructions."""
    monkeypatch.setenv("MONONET_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(
        SOURCES,
        "_fake_script",
        DataSource(
            name="_fake_script",
            hosting="script",
            license="Kaggle ToS",
            url="https://kaggle.com/x",
            prep_hint="run prepare/_fake_script.py",
        ),
    )
    with pytest.raises(FileNotFoundError, match=r"prepare/_fake_script\.py"):
        require_dataset("_fake_script")
