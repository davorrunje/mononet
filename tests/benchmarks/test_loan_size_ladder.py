from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

pytest.importorskip("optuna")
pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks.loan_size_ladder_run import _require_large, main, run_ladder

if TYPE_CHECKING:
    from pathlib import Path


def _synthetic_bundle(n: int = 600, *, name: str = "synthetic") -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, 4))
    # monotone-ish label: increasing in col 0/1, decreasing in col 2
    logit = x[:, 0] + x[:, 1] - x[:, 2]
    y = (logit > 0).astype(np.float64)
    xt = rng.standard_normal((120, 4))
    yt = ((xt[:, 0] + xt[:, 1] - xt[:, 2]) > 0).astype(np.float64)
    return DatasetBundle(
        name=name,
        task="binary_classification",
        X_train=x,
        y_train=y,
        X_test=xt,
        y_test=yt,
        mono_increasing=(0, 1),
        mono_decreasing=(2,),
        feature_names=("a", "b", "c", "d"),
        metadata={},
    )


def test_run_ladder_smoke() -> None:
    """run_ladder returns one finite-IQM record per (n, arm) on a tiny bundle."""
    recs = run_ladder(
        _synthetic_bundle(),
        ns=(100, 400),
        arms=("shallow", "deep"),
        n_trials=2,
        search_seeds=1,
        final_seeds=range(2),
        epochs=1,
    )
    assert len(recs) == 4  # 2 ns x 2 arms
    for r in recs:
        assert r["dataset"] == "synthetic"
        assert r["n"] in (100, 400)
        assert r["arm"] in ("shallow", "deep")
        assert r["test_metric"] == "roc_auc"  # classification -> roc_auc, not accuracy
        assert np.isfinite(r["test_iqm"])
        assert len(r["test_values"]) == 2


def test_require_large_raises_for_small_dataset() -> None:
    """A dataset with n_train < 20_000 cannot be size-laddered."""
    with pytest.raises(ValueError, match="n_train"):
        _require_large(_synthetic_bundle(n=100, name="heart"), "heart")


def test_require_large_accepts_large_dataset() -> None:
    """A dataset with n_train >= 20_000 passes the guard (no raise)."""
    _require_large(_synthetic_bundle(n=25_000, name="lc"), "lc")


def test_run_ladder_guards_small_dataset_when_named() -> None:
    """run_ladder(dataset=...) enforces the >=20k guard at the function boundary."""
    with pytest.raises(ValueError, match="n_train"):
        run_ladder(
            _synthetic_bundle(n=600, name="heart"),
            dataset="heart",
            ns=(100,),
            arms=("shallow",),
            n_trials=2,
            search_seeds=1,
            final_seeds=range(2),
            epochs=1,
        )


def test_main_threads_dataset_into_load_and_run_ladder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--dataset lc is threaded into registry.load and run_ladder (no real training)."""
    calls: dict[str, Any] = {}

    def fake_load(name: str, *, data_dir: Any) -> DatasetBundle:
        calls["load_name"] = name
        return _synthetic_bundle(n=25_000, name=name)

    def fake_run_ladder(bundle: DatasetBundle, **kwargs: Any) -> list[dict[str, Any]]:
        calls["bundle_name"] = bundle.name
        calls["dataset_kwarg"] = kwargs.get("dataset")
        return []

    monkeypatch.setattr("benchmarks.datasets.registry.load", fake_load)
    monkeypatch.setattr("benchmarks.loan_size_ladder_run.run_ladder", fake_run_ladder)
    out = tmp_path / "out.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--dataset",
            "lc",
            "--out",
            str(out),
            "--ns",
            "100",
            "--arms",
            "shallow",
        ],
    )

    main()

    assert calls["load_name"] == "lc"
    assert calls["bundle_name"] == "lc"
    assert calls["dataset_kwarg"] == "lc"
    assert out.exists()


def test_main_raises_for_small_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """--dataset heart (n_train < 20_000) raises ValueError before any training."""

    def fake_load(name: str, *, data_dir: Any) -> DatasetBundle:
        return _synthetic_bundle(n=100, name=name)

    monkeypatch.setattr("benchmarks.datasets.registry.load", fake_load)
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", "heart"])

    with pytest.raises(ValueError, match="n_train"):
        main()
