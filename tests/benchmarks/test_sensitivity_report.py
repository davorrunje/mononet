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
