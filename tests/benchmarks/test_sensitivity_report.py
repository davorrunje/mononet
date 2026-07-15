from typing import Any

import pytest

pytest.importorskip("optuna")
from benchmarks._common.sensitivity_report import best_so_far, saturation_trial


def test_best_so_far_minimize_is_cumulative_min() -> None:
    assert best_so_far([5.0, 3.0, 4.0, 1.0, 2.0], lower=True) == [
        5.0,
        3.0,
        3.0,
        1.0,
        1.0,
    ]


def test_best_so_far_maximize_is_cumulative_max() -> None:
    assert best_so_far([0.1, 0.3, 0.2, 0.9], lower=False) == [0.1, 0.3, 0.3, 0.9]


def test_saturation_trial_reaches_fraction_of_gain() -> None:
    # gain 5->1 = 4; 99% gain = reach <= 1 + 0.01*4 = 1.04; first at 1-based 4.
    traj = [5.0, 3.0, 3.0, 1.0, 1.0]
    assert saturation_trial(traj, lower=True, p=0.99) == 4


def test_saturation_trial_flat_trajectory_is_one() -> None:
    assert saturation_trial([2.0, 2.0, 2.0], lower=True) == 1


from benchmarks._common.sensitivity_report import (  # noqa: E402
    completed_values,
    incumbent_changepoints,
)


class _FakeTrial:
    def __init__(
        self, value: float | None, params: dict[str, Any], complete: bool = True
    ) -> None:
        import optuna

        self.value = value
        self.params = params
        self.state = (
            optuna.trial.TrialState.COMPLETE
            if complete
            else optuna.trial.TrialState.FAIL
        )
        self.user_attrs: dict[str, Any] = {}


class _FakeStudy:
    def __init__(self, trials: list[_FakeTrial]) -> None:
        self.trials = trials


def test_completed_values_skips_non_complete() -> None:
    s = _FakeStudy([_FakeTrial(1.0, {}), _FakeTrial(None, {}, complete=False)])
    assert completed_values(s, lower=True) == [1.0]


def test_incumbent_changepoints_are_the_improving_trials() -> None:
    s = _FakeStudy(
        [
            _FakeTrial(5.0, {"depth": 1}),
            _FakeTrial(3.0, {"depth": 2}),
            _FakeTrial(4.0, {"depth": 3}),
            _FakeTrial(1.0, {"depth": 4}),
        ]
    )
    cps = incumbent_changepoints(s, lower=True)
    assert [i for i, _ in cps] == [1, 2, 4]
    assert cps[-1][1] == {"depth": 4}


import benchmarks._common.sensitivity_report as sr  # noqa: E402


def test_incumbent_test_curve_reevaluates_once_per_incumbent(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from types import SimpleNamespace

    study = _FakeStudy(
        [
            _FakeTrial(5.0, {"depth": 1}),
            _FakeTrial(3.0, {"depth": 2}),
            _FakeTrial(4.0, {"depth": 3}),
        ]
    )
    calls = {"n": 0}

    def fake_final_eval(bundle: Any, params: dict[str, Any], **kw: Any) -> Any:
        calls["n"] += 1
        return SimpleNamespace(metric=float(params["depth"])), []

    monkeypatch.setattr(sr, "final_eval", fake_final_eval)
    curve, n_eval = sr.incumbent_test_curve(
        study,
        bundle=object(),
        mode="split",
        residual=False,
        backend="torch",
        lower=True,
        n_trials=3,
        seeds=range(1),
    )
    # incumbents at trials 1 (depth1) and 2 (depth2); trial 3 holds depth2.
    assert curve == [1.0, 2.0, 2.0]
    assert n_eval == 2
    assert calls["n"] == 2


def test_render_plot_writes_png_and_pdf(tmp_path) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("matplotlib")
    series = {
        "heart": {
            "split-plain": ([0.9, 0.91, 0.91], [0.88, 0.89, 0.89]),
            "alternate-plain": ([0.89, 0.90, 0.906], None),
        }
    }
    out = tmp_path / "sensitivity"
    sr.render_plot(series, out)
    assert out.with_suffix(".png").stat().st_size > 0
    assert out.with_suffix(".pdf").stat().st_size > 0
