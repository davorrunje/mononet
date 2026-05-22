"""Contract test for the mononet.torch public API surface."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_mono_linear_exists_and_is_nn_module() -> None:
    from mononet.torch import MonoLinear

    assert issubclass(MonoLinear, torch.nn.Module)


def test_mono_mlp_exists_and_is_nn_module() -> None:
    from mononet.torch import MonoMLP

    assert issubclass(MonoMLP, torch.nn.Module)


def test_instantiating_mono_linear_raises_not_implemented() -> None:
    import numpy as np

    from mononet.core.types import ActivationSpec, MonotonicityMask
    from mononet.torch import MonoLinear

    with pytest.raises(NotImplementedError):
        MonoLinear(
            in_features=4,
            out_features=2,
            monotonicity=MonotonicityMask(np.zeros(4, dtype=np.int8)),
            activation=ActivationSpec(name="relu"),
        )


def test_no_unexpected_top_level_exports() -> None:
    import mononet.torch as t

    expected = {"MonoLinear", "MonoMLP"}
    actual = set(t.__all__)
    assert actual == expected, f"got: {actual}"
