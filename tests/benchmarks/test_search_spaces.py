from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

optuna = pytest.importorskip("optuna")

from benchmarks._common.search_spaces import suggest_config  # noqa: E402

if TYPE_CHECKING:
    from benchmarks._common.config import BenchmarkConfig


def _cfg(
    mode: Literal["split", "mixed"],
    residual: bool,
    metric: Literal["accuracy", "rmse", "mse", "roc_auc"] = "mse",
    deep: bool = False,
    n_train: int = 10_000,
) -> BenchmarkConfig:
    study = optuna.create_study()
    trial = study.ask()
    return suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode=mode,
        residual=residual,
        epochs=3,
        metric=metric,
        n_train=n_train,
        deep=deep,
    )


def test_mixed_searches_convex_fraction_within_unit_interval() -> None:
    cfg = _cfg("mixed", False, metric="mse")
    assert cfg.mode == "mixed"
    assert cfg.residual is False
    assert 0.0 <= cfg.convex_fraction <= 1.0
    assert cfg.activation == "elu"
    assert cfg.epochs == 3
    assert 1 <= cfg.depth <= 4
    assert cfg.metrics == ("mse",)


def test_split_uses_fixed_convex_fraction() -> None:
    # split mode ignores convex_fraction; the sampler must NOT add it as a
    # search dimension (kept at the 0.5 default so studies don't carry a dead param).
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode="split",
        residual=True,
        epochs=2,
        metric="accuracy",
        n_train=10_000,
    )
    assert cfg.mode == "split"
    assert cfg.residual is True
    assert cfg.convex_fraction == 0.5
    assert "convex_fraction" not in trial.params
    assert cfg.metrics == ("accuracy",)


def test_roc_auc_primary_also_reports_accuracy() -> None:
    # When roc_auc is the search objective, accuracy must still be reported
    # alongside it (the primary metric switched away from accuracy, but
    # accuracy is not dropped from the results).
    cfg = _cfg("split", residual=True, metric="roc_auc")
    assert cfg.metrics == ("roc_auc", "accuracy")


def test_deep_samples_depth_from_high_band() -> None:
    # Deep flavor draws depth from the categorical high band, never the 1..4 range.
    for _ in range(25):
        cfg = _cfg("mixed", residual=True, deep=True)
        assert cfg.depth in (6, 10, 16)
        assert cfg.residual is True
        assert 0.0 <= cfg.convex_fraction <= 1.0  # other fields still sampled


def test_non_deep_keeps_shallow_depth_band() -> None:
    for _ in range(25):
        cfg = _cfg("split", residual=True, deep=False)
        assert 1 <= cfg.depth <= 4


def _cfg_for_dataset(dataset: str, n_train: int | None = None) -> BenchmarkConfig:
    if n_train is None:
        # Map dataset name to a representative n_train value.
        # Large datasets (loan, blog) are >= 20_000; small datasets are < 20_000.
        n_train = 50_000 if dataset in ("loan", "blog") else 5_000
    study = optuna.create_study()
    trial = study.ask()
    return suggest_config(
        trial,
        dataset=dataset,
        backend="torch",
        mode="split",
        residual=False,
        epochs=3,
        metric="mse",
        n_train=n_train,
    )


def test_small_datasets_use_standard_batch_band() -> None:
    for _ in range(25):
        cfg = _cfg_for_dataset("auto")
        assert cfg.batch_size in (8, 16, 32, 64, 128, 256)


def test_large_datasets_use_large_batch_band() -> None:
    # loan/blog are large enough that tiny batches make training intractable;
    # the sampler must draw only from the large-batch band.
    for dataset in ("loan", "blog"):
        for _ in range(25):
            cfg = _cfg_for_dataset(dataset)
            assert cfg.batch_size in (512, 1024, 2048, 4096)


def test_batch_band_is_size_driven() -> None:
    """Batch band is selected by train-set size, not dataset name."""
    from benchmarks._common.search_spaces import (
        _BATCH_SIZES_LARGE,
        _BATCH_SIZES_SMALL,
    )

    def band(n_train: int) -> list[int]:
        seen: set[int] = set()
        study = optuna.create_study()
        for _ in range(60):
            t = study.ask()
            cfg = suggest_config(
                t,
                dataset="x",
                backend="torch",
                mode="mixed",
                residual=True,
                epochs=1,
                metric="mse",
                n_train=n_train,
            )
            seen.add(cfg.batch_size)
        return sorted(seen)

    assert set(band(50_000)).issubset(set(_BATCH_SIZES_LARGE))
    assert set(band(500)).issubset(set(_BATCH_SIZES_SMALL))


def test_alternate_sets_composition_init_and_no_convex_fraction() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode="alternate",
        residual=False,
        epochs=3,
        metric="mse",
        n_train=10_000,
    )
    assert cfg.mode == "alternate"
    assert cfg.alt_init == "composition"
    assert cfg.convex_fraction == 0.5  # not a search dim for alternate
    assert "convex_fraction" not in trial.params


def test_search_activation_samples_one_of_four() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode="mixed",
        residual=False,
        epochs=3,
        metric="mse",
        n_train=10_000,
        search_activation=True,
    )
    assert cfg.activation in ("relu", "elu", "softplus", "selu")
    assert "activation" in trial.params


def test_default_activation_is_elu_and_alt_init_none() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode="mixed",
        residual=False,
        epochs=3,
        metric="mse",
        n_train=10_000,
    )
    assert cfg.activation == "elu"
    assert cfg.alt_init is None
    assert "activation" not in trial.params


def test_embed_layers_controls_embedding_depth() -> None:
    study = optuna.create_study()
    trial = study.ask()
    cfg = suggest_config(
        trial,
        dataset="syn",
        backend="torch",
        mode="mixed",
        residual=False,
        epochs=3,
        metric="mse",
        n_train=10_000,
        embed_layers=2,
    )
    assert cfg.embed_hidden == (cfg.width, cfg.width)


def test_max_depth_caps_depth() -> None:
    for _ in range(25):
        study = optuna.create_study()
        trial = study.ask()
        cfg = suggest_config(
            trial,
            dataset="syn",
            backend="torch",
            mode="mixed",
            residual=False,
            epochs=3,
            metric="mse",
            n_train=10_000,
            max_depth=3,
        )
        assert 1 <= cfg.depth <= 3
