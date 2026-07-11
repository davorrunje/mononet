from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_data_dir_is_lfs_tracked() -> None:
    """Any file under benchmarks/data/ resolves to the git-lfs filter."""
    out = subprocess.run(
        [
            "git",
            "check-attr",
            "filter",
            "--",
            "benchmarks/data/adult/train_adult.csv.gz",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "filter: lfs" in out


def test_devcontainer_installs_git_lfs() -> None:
    """The shared devcontainer setup installs and initialises git-lfs."""
    script = (REPO / ".devcontainer/shared/install_common_tools.sh").read_text()
    assert "git-lfs" in script
    assert "git lfs install" in script
    assert "git lfs pull" in script
