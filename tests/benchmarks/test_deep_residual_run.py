import pytest

pytest.importorskip("torch")

import torch.nn as nn

from benchmarks._common.init_diagnostics import build_residual_stack
from mononet.torch import MonoResidual


def test_build_residual_stack_block_count() -> None:
    net = build_residual_stack("mixed", depth=8, sub_depth=2, width=16)
    assert sum(isinstance(m, MonoResidual) for m in net.children()) == 4  # 8 // 2


def test_build_residual_stack_plain_has_no_residual() -> None:
    net = build_residual_stack("mixed", depth=8, sub_depth=None, width=16)
    assert sum(isinstance(m, MonoResidual) for m in net.children()) == 0
    assert isinstance(net, nn.Sequential)
