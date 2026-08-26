# SPDX-License-Identifier: Apache-2.0
"""Guard: every tool that targets a Python version agrees with `requires-python`.

`[project] requires-python` is the single source of truth for which Python
versions this package supports. Tools that must not emit or accept code the
lowest supported version cannot run have to track its floor, and each one that
hardcodes a version instead is a place the floor can silently drift.

`ruff` needs no assertion here: with `[tool.ruff] target-version` unset it infers
the floor from `requires-python` itself. This module checks the settings that
cannot infer it, and fails if `target-version` is ever reintroduced with a value
that disagrees.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"


def _pyproject() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        data: dict[str, Any] = tomllib.load(handle)
    return data


def _floor() -> tuple[int, int]:
    """Return the lowest supported version as `(major, minor)`."""
    requires = str(_pyproject()["project"]["requires-python"])
    lower = next(part for part in requires.split(",") if part.strip().startswith(">="))
    major, minor = lower.strip().removeprefix(">=").strip().split(".")[:2]
    return int(major), int(minor)


def _hook_args(hook_id: str) -> Iterator[str]:
    config = yaml.safe_load(PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    for repo in config["repos"]:
        for hook in repo["hooks"]:
            if hook["id"] == hook_id:
                yield from hook.get("args", [])


def test_pyupgrade_targets_the_floor_of_requires_python() -> None:
    major, minor = _floor()
    expected = f"--py{major}{minor}-plus"
    args = list(_hook_args("pyupgrade"))
    assert expected in args, (
        f"pyupgrade must target the floor of requires-python ({expected}), "
        f"got {args}. A higher target lets it rewrite source into syntax that "
        f"Python {major}.{minor} cannot run."
    )


def test_ruff_target_version_is_inferred_or_matches_the_floor() -> None:
    major, minor = _floor()
    configured = _pyproject().get("tool", {}).get("ruff", {}).get("target-version")
    assert configured in (None, f"py{major}{minor}"), (
        f"[tool.ruff] target-version is {configured!r} but the floor of "
        f"requires-python is py{major}{minor}. Either drop the setting so ruff "
        f"infers it, or correct it."
    )
