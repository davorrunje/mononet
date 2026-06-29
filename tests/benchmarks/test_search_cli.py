import pytest

pytest.importorskip("typer")
pytest.importorskip("optuna")

from typer.testing import CliRunner

from benchmarks.search import _SMOKE, app

runner = CliRunner()


def test_help_lists_flags() -> None:
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    for flag in (
        "--datasets",
        "--flavors",
        "--backend",
        "--n-trials",
        "--n-jobs",
        "--smoke",
        "--dry-run",
    ):
        assert flag in res.output


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
