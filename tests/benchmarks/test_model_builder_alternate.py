"""Builder support for the ``alternate`` construction (composition + legacy)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from benchmarks._common.bundle import DatasetBundle
from benchmarks._common.config import BenchmarkConfig, OptimizerSpec


def _bundle() -> DatasetBundle:
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (64, 3)).astype("float32")
    y = x.sum(1).astype("float32")
    return DatasetBundle(
        name="t",
        task="regression",
        X_train=x,
        y_train=y,
        X_test=x,
        y_test=y,
        mono_increasing=(0, 1, 2),
        mono_decreasing=(),
        feature_names=("a", "b", "c"),
        metadata={},
    )


def _cfg(mode: str, alt_init: str | None, depth: int = 4) -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset="t",
        backend="torch",
        mode=mode,  # type: ignore[arg-type]
        residual=False,
        depth=depth,
        width=16,
        activation="relu",
        convex_fraction=0.5,
        embed_hidden=(),
        dropout=0.0,
        optimizer=OptimizerSpec(name="adam", lr=1e-3),
        lr_decay=None,
        batch_size=64,
        epochs=1,
        early_stopping=None,
        seeds=(0,),
        metrics=("mse",),
        alt_init=alt_init,  # type: ignore[arg-type]
    )


def test_alternate_composition_builds_and_is_finite() -> None:
    import torch

    from benchmarks._common.model_builder import build_model

    m = build_model(_cfg("alternate", "composition"), _bundle())
    out = m(torch.zeros(2, 3))
    assert torch.isfinite(out).all()


def test_alternate_legacy_builds_pure_mixed_layers() -> None:
    import torch

    from benchmarks._common.model_builder import build_model

    m = build_model(_cfg("alternate", "legacy"), _bundle())
    # legacy arm uses mode="mixed" pure-class layers (convex_fraction 1/0 alternating)
    pure = [
        mod
        for mod in m.modules()
        if getattr(mod, "mode", None) == "mixed"
        and float(getattr(mod, "convex_fraction", -1.0)) in (0.0, 1.0)
    ]
    assert len(pure) >= 4  # the 4 alternating stack layers
    assert torch.isfinite(m(torch.zeros(2, 3))).all()


def test_alternate_stack_uses_alternate_layers_for_composition() -> None:
    from benchmarks._common.model_builder import build_model

    m = build_model(_cfg("alternate", "composition"), _bundle())
    alt = [mod for mod in m.modules() if getattr(mod, "mode", None) == "alternate"]
    assert len(alt) >= 4  # the stack is built from alternate layers


def test_residual_alternate_not_supported() -> None:
    from benchmarks._common.model_builder import build_model

    cfg = _cfg("alternate", "composition").replace(residual=True)
    with pytest.raises(NotImplementedError, match="residual"):
        build_model(cfg, _bundle())
