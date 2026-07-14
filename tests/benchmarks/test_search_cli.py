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
        "--cv-folds",
        "--smoke",
        "--dry-run",
    ):
        assert flag in output


def test_smoke_preset_values() -> None:
    assert _SMOKE["datasets"] == ["auto", "heart"]
    assert _SMOKE["n_trials"] == 5
    assert _SMOKE["epochs"] == 5
    assert _SMOKE["cv_folds"] == 2
    assert "final_top_k" not in _SMOKE


def test_dry_run_reports_plan_without_running() -> None:
    res = runner.invoke(app, ["--smoke", "--dry-run"])
    assert res.exit_code == 0
    assert "auto" in res.output
    assert "heart" in res.output
    assert "would run" in res.output.lower()


def test_invalid_flavors_exits_nonzero() -> None:
    res = runner.invoke(app, ["--flavors", "foo", "--dry-run"])
    assert res.exit_code != 0


def test_parse_flavors_accepts_deep() -> None:
    from benchmarks.search import _parse_flavors

    assert _parse_flavors("mixed-deep") == (("mixed", True, True),)
    assert _parse_flavors("split-plain,split-residual") == (
        ("split", False, False),
        ("split", True, False),
    )


def test_parse_flavors_default_is_all_six() -> None:
    from benchmarks.search import _parse_flavors

    assert len(_parse_flavors("")) == 6


def test_dry_run_lists_deep_flavor() -> None:
    res = runner.invoke(app, ["--flavors", "mixed-deep", "--dry-run"])
    assert res.exit_code == 0
    assert "mixed-deep" in res.output


def test_dry_run_default_datasets_includes_all_22() -> None:
    """Verify default --datasets (when unset) covers all 10 real + 12 synthetic."""
    res = runner.invoke(app, ["--dry-run"])
    assert res.exit_code == 0
    output = res.output

    # 10 real datasets
    real = [
        "auto",
        "heart",
        "compas",
        "loan",
        "blog",
        "adult",
        "taiwan",
        "polish",
        "german",
        "lc",
    ]
    for name in real:
        assert name in output, f"missing real dataset: {name}"

    # 12 synthetic datasets
    synth = [
        "synth_additive_clow",
        "synth_additive_cmid",
        "synth_additive_chigh",
        "synth_teacher_relu_clow",
        "synth_teacher_relu_cmid",
        "synth_teacher_relu_chigh",
        "synth_teacher_elu_clow",
        "synth_teacher_elu_cmid",
        "synth_teacher_elu_chigh",
        "synth_lattice_clow",
        "synth_lattice_cmid",
        "synth_lattice_chigh",
    ]
    for name in synth:
        assert name in output, f"missing synth dataset: {name}"


def test_parse_flavors_accepts_alternate_plain() -> None:
    from benchmarks.search import _parse_flavors

    assert _parse_flavors("alternate-plain") == (("alternate", False, False),)


def test_parse_flavors_rejects_alternate_residual() -> None:
    import typer

    from benchmarks.search import _parse_flavors

    with pytest.raises(typer.BadParameter):
        _parse_flavors("alternate-residual")
