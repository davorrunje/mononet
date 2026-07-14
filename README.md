# mononet — Constrained Monotonic Neural Networks

[![PyPI version](https://img.shields.io/pypi/v/mononet)](https://pypi.org/project/mononet/)
[![Python versions](https://img.shields.io/pypi/pyversions/mononet)](https://pypi.org/project/mononet/)
[![License](https://img.shields.io/pypi/l/mononet)](https://github.com/davorrunje/mononet/blob/main/LICENSE)
[![codecov](https://codecov.io/gh/davorrunje/mononet/graph/badge.svg)](https://codecov.io/gh/davorrunje/mononet)
[![Build](https://github.com/davorrunje/mononet/actions/workflows/build.yml/badge.svg)](https://github.com/davorrunje/mononet/actions/workflows/build.yml)
[![Docs](https://img.shields.io/badge/docs-mononet-blue)](https://davorrunje.github.io/mononet/)
[![arXiv](https://img.shields.io/badge/arXiv-2205.11775-b31b1b.svg)](https://arxiv.org/abs/2205.11775)

Multi-backend implementation of the constrained monotonic neural network
construction from:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

with the optional activation-split refinement (`mode="split"`) from:

> Sartor, D. et al. (2025). *Advancing Constrained Monotonic Neural
> Networks.* ICML 2025. <https://arxiv.org/abs/2505.02537>

First-class support for **PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

> **CPU-only torch:** on linux the `torch`/`all` extras pull PyTorch's default
> CUDA wheel. Under **uv**, use the `all-cpu` (or `torch-cpu`) extra for a
> CUDA-free install. Plain `pip` cannot force CPU torch via an extra — see the
> [installation docs](https://davorrunje.github.io/mononet/installation.html).
> The `default` devcontainer already uses `all-cpu`.

## Quick start

`mononet` ships **layers**, not composed models — stack them with your
framework's native `Sequential` (or equivalent). Each backend exposes
`MonoResidual`, `MonoInput`, and the framework-idiomatic dense layer:
`MonoLinear` for PyTorch and JAX, `MonoDense` for Keras.

A mixed-feature example: monotone in 3 features (2 non-decreasing, 1
non-increasing) via `MonoInput`, and unconstrained in 2 non-monotone
features, which are embedded through a plain MLP. `MonoLinear` and
`MonoResidual` default to `mode="mixed"`.

```python
"""Mixed-feature monotone network (PyTorch).

Monotone in 3 features (2 non-decreasing, 1 non-increasing) via ``MonoInput``,
and unconstrained in 2 non-monotone features, which are embedded through a
plain MLP. The embedding absorbs the non-monotonicity, so the composite map is
monotone in ``x_mono`` and free in ``x_free``. Mixed mode is the default.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mononet import MonotonicityMask
from mononet.torch import MonoInput, MonoLinear, MonoResidual


class RiskNet(nn.Module):
    """Monotone in ``x_mono`` (directions +1, +1, -1); free in ``x_free``."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.mono_in = MonoInput(MonotonicityMask(np.array([1, 1, -1], dtype=np.int8)))
        self.net = nn.Sequential(
            MonoLinear(11, 64, activation="elu"),
            MonoResidual(64, 64, activation="elu"),
            MonoResidual(64, 64, activation="elu"),
            MonoLinear(64, 1),
        )

    def forward(self, x_mono: torch.Tensor, x_free: torch.Tensor) -> torch.Tensor:
        """Combine the sign-flipped monotone features with the free embedding."""
        z = torch.cat([self.mono_in(x_mono), self.embed(x_free)], dim=-1)
        return self.net(z)
```

For per-feature monotonicity directions, pass a
`mononet.core.types.MonotonicityMask` (a 1-D array of `{-1, +1}`) to
`MonoInput`. The same layers exist under `mononet.jax` and
`mononet.keras`; see the [per-backend guides](docs/guides/).

## Benchmark results

Held-out accuracy on the paper's five tabular datasets, comparing the `split`
and `mixed` monotone constructions at shallow (`plain`) and deep (`residual`)
depth. Cells report **IQM** (interquartile mean; robust) and **mean ± std** over
seeds, with the effective monotone-layer count `L` and a collapse flag `⚠` (shown
only when some seeds degenerated). Metric per dataset: MSE (`auto`), RMSE
(`blog`), accuracy (`heart`/`compas`/`loan`); **↓** lower / **↑** higher is
better. **Bold** = best per dataset. Full methodology and the per-flavor
robustness table are in the
[benchmark docs](https://davorrunje.github.io/mononet/benchmarks/deep-residual-accuracy.html).

| dataset | mode | variant | layers | IQM | mean ± std | ⚠ |
|---|---|---|--:|--:|--:|:-:|
| auto (MSE ↓) | split | plain | 2 | **9.78** | 9.76 ± 0.18 | · |
|  | split | residual | 4 | 9.89 | 10.11 ± 0.62 | 2/20 |
|  | mixed | plain | 2 | 10.91 | 10.90 ± 0.21 | · |
|  | mixed | residual | 4 | 9.92 | 9.94 ± 0.33 | · |
| heart (acc ↑) | split | plain | 4 | 0.836 | 0.711 ± 0.249 | 4/20 |
|  | split | residual | 14 | 0.831 | 0.829 ± 0.012 | 2/20 |
|  | mixed | plain | 3 | **0.836** | 0.839 ± 0.012 | · |
|  | mixed | residual | 4 | 0.821 | 0.825 ± 0.008 | · |
| compas (acc ↑) | split | plain | 2 | 0.679 | 0.679 ± 0.002 | · |
|  | split | residual | 14 | 0.641 | 0.632 ± 0.033 | 4/20 |
|  | mixed | plain | 4 | 0.683 | 0.683 ± 0.002 | · |
|  | mixed | residual | 10 | **0.684** | 0.684 ± 0.002 | · |
| loan (acc ↑) | split | plain | 3 | 0.647 | 0.647 ± 0.001 | · |
|  | split | residual | 6 | 0.647 | 0.646 ± 0.001 | · |
|  | mixed | plain | 3 | 0.648 | 0.648 ± 0.000 | · |
|  | mixed | residual | 14 | **0.649** | 0.650 ± 0.001 | · |
| blog (RMSE ↓) | split | plain | 2 | 0.185 | 0.185 ± 0.002 | · |
|  | split | residual | 4 | 0.182 | 0.182 ± 0.000 | 1/10 |
|  | mixed | plain | 2 | 0.189 | 0.189 ± 0.000 | · |
|  | mixed | residual | 4 | **0.173** | 0.173 ± 0.001 | · |

`residual` collapses the better of the residual/deep depth bands (by CV);
`L = 2·blocks + 2` effective monotone layers. Deep `mixed residual` is
nominally best on `loan` (the largest dataset) above, but a controlled
[size-ladder study](https://davorrunje.github.io/mononet/benchmarks/loan-size-ladder.html)
— deep vs shallow *residual*, tuned independently at each training-set size —
finds that edge is within noise and does not grow with scale, so **depth is
neutral even on `loan`**; elsewhere ≤ 4 layers is best. `mixed` wins 4 of 5
datasets; the `⚠` instabilities are all shallow `split`.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
Commercial use is permitted. The technique is described in U.S. Patent
11,551,063 (assignee: AIRT Technologies Ltd.); the Apache-2.0 license
grants the patent rights needed to use this code. For academic use, please
cite the paper (see [`NOTICE.md`](NOTICE.md)).

## Formal proofs

The theory underpinning `mononet` is mechanized in Lean 4 + mathlib4
(`sorry`-free) in the companion repo
**[neural-network-proofs](https://github.com/davorrunje/neural-network-proofs)** —
browse the proofs, blueprint, and API docs at
<https://davorrunje.github.io/neural-network-proofs/>.

## Documentation

Full docs at <https://davorrunje.github.io/mononet/>. Source for guides
and benchmarks lives in [`docs/`](docs/).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow:
devcontainer choice, `uv sync`, pre-commit, per-backend test commands.

## Citation

If you use `mononet` in academic work, please cite the paper:

```bibtex
@inproceedings{runje2023constrained,
  title         = {Constrained Monotonic Neural Networks},
  author        = {Runje, Davor and Shankaranarayana, Sharath M.},
  booktitle     = {Proceedings of the 40th International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {202},
  year          = {2023},
  publisher     = {PMLR},
  url           = {https://proceedings.mlr.press/v202/runje23a.html},
  eprint        = {2205.11775},
  archivePrefix = {arXiv}
}
```
