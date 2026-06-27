from benchmarks._common.config import BenchmarkConfig, OptimizerSpec


def _cfg() -> BenchmarkConfig:
    return BenchmarkConfig(
        dataset="auto", backend="torch", mode="switch", residual=False,
        depth=2, width=21, activation="elu", convex_fraction=0.5,
        embed_hidden=(21,), dropout=0.0,
        optimizer=OptimizerSpec(name="adam", lr=1e-3, weight_decay=0.0),
        lr_decay=None, batch_size=16, epochs=50, early_stopping=None,
        seeds=(0, 1), metrics=("mse",),
    )


def test_replace_returns_modified_copy():
    c = _cfg()
    d = c.replace(mode="absolute", residual=True)
    assert c.mode == "switch"
    assert c.residual is False
    assert d.mode == "absolute"
    assert d.residual is True
    assert d.dataset == "auto"
