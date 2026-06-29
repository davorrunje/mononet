import re

import pytest

pytest.importorskip("typer")
pytest.importorskip("optuna")

from typer.testing import CliRunner

from benchmarks.search import _SMOKE, app

runner = CliRunner()

# Typer's Rich help renderer colorizes each option name with ANSI spans
# (e.g. "-\x1b[..m-datasets") whenever the runner reports a color terminal,
# which splits the literal flag. Strip ANSI before substring assertions.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def test_help_lists_flags() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    output = _ANSI.sub("", res.output)
    for flag in (
        "--datasets",
        "--flavors",
        "--backend",
        "--n-trials",
        "--n-jobs",
        "--smoke",
        "--dry-run",
    ):
        assert flag in output


def test_smoke_preset_values() -> None:
    assert _SMOKE["datasets"] == ["auto", "heart"]
    assert _SMOKE["n_trials"] == 5
    assert _SMOKE["epochs"] == 5


def test_dry_run_reports_plan_without_running() -> None:
    res = runner.invoke(app, ["--smoke", "--dry-run"])
    assert res.exit_code == 0
    assert "auto" in res.output
    assert "heart" in res.output
    assert "would run" in res.output.lower()


def test_invalid_flavors_exits_nonzero() -> None:
    res = runner.invoke(app, ["--flavors", "foo", "--dry-run"])
    assert res.exit_code != 0
