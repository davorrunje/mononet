from collections.abc import Sequence

import numpy as np

from benchmarks._common.results import (
    ResultRow,
    aggregate,
    interquartile_mean,
)


def _rows(values: Sequence[float]) -> list[ResultRow]:
    return [
        ResultRow(
            dataset="auto",
            backend="torch",
            mode="split",
            residual=False,
            seed=i,
            scores={"mse": v},
            epochs_run=50,
        )
        for i, v in enumerate(values)
    ]


def test_best_5_of_10_takes_lowest_for_loss() -> None:
    # 10 seeds; "mse" is lower-is-better, so best 5 = the 5 smallest.
    rows = _rows([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    agg = aggregate(rows, metric="mse", lower_is_better=True, top_k=5)
    assert agg.n_seeds == 10
    assert agg.n_selected == 5
    assert np.isclose(agg.mean, np.mean([1, 2, 3, 4, 5]))
    assert np.isclose(agg.std, np.std([1, 2, 3, 4, 5]))


def test_best_5_of_10_takes_highest_for_accuracy() -> None:
    rows = _rows([0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
    agg = aggregate(rows, metric="mse", lower_is_better=False, top_k=5)
    assert np.isclose(agg.mean, np.mean([0.55, 0.56, 0.57, 0.58, 0.59]))


def test_interquartile_mean_drops_extremes() -> None:
    # Two collapsed runs (0.2) and one lucky run (0.99) among stable ~0.83s;
    # IQM should ignore the extremes and land near the stable cluster.
    vals = np.array([0.2, 0.2, 0.82, 0.83, 0.83, 0.84, 0.84, 0.99])
    iqm = interquartile_mean(vals)
    assert 0.82 <= iqm <= 0.85
    # mean is dragged down by the collapses; IQM is not.
    assert iqm > float(vals.mean())


def test_aggregate_reports_robust_protocols_over_all_rows() -> None:
    # 8 seeds, two collapsed. mean/std (top_k=all) + median + iqm + per-seed.
    rows = _rows([0.2, 0.2, 0.82, 0.83, 0.83, 0.84, 0.84, 0.99])
    agg = aggregate(rows, metric="mse", lower_is_better=False, top_k=len(rows))
    assert agg.n_seeds == 8
    assert len(agg.values) == 8
    assert np.isclose(agg.median, 0.83)
    assert 0.82 <= agg.iqm <= 0.85
    assert np.isclose(agg.mean, float(np.mean(agg.values)))  # top_k=all -> plain mean


def test_count_collapses_classification_and_regression() -> None:
    from benchmarks._common.search import _count_collapses

    # classification (accuracy): base rate 0.79; seeds at/below 0.81 count as
    # collapsed.
    vals = (0.21, 0.21, 0.83, 0.84, 0.84)
    assert (
        _count_collapses(
            vals,
            task="binary_classification",
            base_rate=0.79,
            lower_is_better=False,
            metric="accuracy",
        )
        == 2
    )
    # regression: gross high outlier (bad side) is a collapse.
    reg = (9.5, 9.6, 9.7, 9.8, 640.0)
    assert (
        _count_collapses(
            reg,
            task="regression",
            base_rate=0.0,
            lower_is_better=True,
            metric="mse",
        )
        == 1
    )


def test_count_collapses_is_metric_aware_for_roc_auc() -> None:
    from benchmarks._common.search import _count_collapses

    # roc_auc: chance AUC is 0.5 regardless of imbalance, so the collapse floor
    # is 0.5 + 0.02, NOT base_rate. With an imbalanced base_rate=0.9 the old
    # accuracy rule (base_rate + 0.02) would wrongly flag 0.60 as collapsed.
    near_chance = (0.51, 0.52)
    assert (
        _count_collapses(
            near_chance,
            task="binary_classification",
            base_rate=0.9,
            lower_is_better=False,
            metric="roc_auc",
        )
        == 2
    )
    healthy = (0.60, 0.62)
    assert (
        _count_collapses(
            healthy,
            task="binary_classification",
            base_rate=0.9,
            lower_is_better=False,
            metric="roc_auc",
        )
        == 0
    )
    # sanity: under the accuracy rule the same 0.60/0.62 values WOULD be
    # flagged (base_rate 0.9), confirming the metric argument changes behavior.
    assert (
        _count_collapses(
            healthy,
            task="binary_classification",
            base_rate=0.9,
            lower_is_better=False,
            metric="accuracy",
        )
        == 2
    )
