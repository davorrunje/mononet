# SPDX-License-Identifier: Apache-2.0
"""Tests for `tools/supported_pythons.py`."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "supported_pythons.py"


def _load() -> ModuleType:
    """Import the script by path; `tools/` is not an importable package."""
    spec = importlib.util.spec_from_file_location("supported_pythons", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expand_bounded_range() -> None:
    assert _load().expand(">=3.11,<3.15") == ["3.11", "3.12", "3.13", "3.14"]


def test_expand_excludes_the_upper_bound() -> None:
    assert _load().expand(">=3.11,<3.12") == ["3.11"]


def test_expand_unbounded_runs_to_the_candidate_ceiling() -> None:
    module = _load()
    result = module.expand(">=3.13")
    assert result[0] == "3.13"
    assert result[-1] == module.CANDIDATES[-1]


def test_expand_honours_an_explicit_candidate_list() -> None:
    assert _load().expand(">=3.11,<3.15", ["3.10", "3.12", "3.99"]) == ["3.12"]


def test_read_requires_python_reads_the_project_table(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nrequires-python = ">=3.11,<3.13"\n',
        encoding="utf-8",
    )
    assert _load().read_requires_python(pyproject) == ">=3.11,<3.13"


def test_the_repo_requires_python_expands_to_a_sorted_nonempty_list() -> None:
    module = _load()
    versions = module.expand(module.read_requires_python())
    assert versions, "requires-python expanded to nothing"
    minors = [int(v.split(".")[1]) for v in versions]
    assert minors == sorted(minors)


def test_cli_emits_one_version_per_line() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--requires-python", ">=3.11,<3.14"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.split() == ["3.11", "3.12", "3.13"]


def test_cli_json_is_parseable_by_the_ci_matrix() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--json", "--requires-python", ">=3.11,<3.13"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["3.11", "3.12"]
