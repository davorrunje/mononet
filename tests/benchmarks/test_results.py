import numpy as np

from benchmarks._common.results import ResultRow, aggregate


def _rows(values):
    return [
        ResultRow(dataset="auto", backend="torch", mode="switch", residual=False,
                  seed=i, scores={"mse": v}, epochs_run=50)
        for i, v in enumerate(values)
    ]


def test_best_5_of_10_takes_lowest_for_loss():
    # 10 seeds; "mse" is lower-is-better, so best 5 = the 5 smallest.
    rows = _rows([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    agg = aggregate(rows, metric="mse", lower_is_better=True, top_k=5)
    assert agg.n_seeds == 10
    assert agg.n_selected == 5
    assert np.isclose(agg.mean, np.mean([1, 2, 3, 4, 5]))
    assert np.isclose(agg.std, np.std([1, 2, 3, 4, 5]))


def test_best_5_of_10_takes_highest_for_accuracy():
    rows = _rows([0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59])
    agg = aggregate(rows, metric="mse", lower_is_better=False, top_k=5)
    assert np.isclose(agg.mean, np.mean([0.55, 0.56, 0.57, 0.58, 0.59]))
