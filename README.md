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

Tuned shallow bake-off on the paper's five tabular datasets, comparing four
monotone constructions at ≤ 4 effective layers (`plain`, `depth ∈ [1,3]`): the
`split` activation-switch, `mixed` (the `|W|` constraint, tuning the
`convex_fraction` split), `mixed-fixed` (same, with `convex_fraction` held at
0.5), and `alternate` (per-layer σ/σ′ alternation). Each flavor is tuned
per-dataset with its own Optuna search (activation included) at the paper's
per-dataset trial counts, under a stability-aware CV objective. Cells report
**IQM** (interquartile mean; robust) and **mean ± std** over seeds, with the
effective monotone-layer count `L`. Metric per dataset: MSE (`auto`), RMSE
(`blog`), accuracy (`heart`/`compas`/`loan`); **↓** lower / **↑** higher is
better. **Bold** 🥇 = best per dataset. Full per-HP table, the ablation, and
bootstrap verdicts are in the
[benchmark docs](https://davorrunje.github.io/mononet/benchmarks/alternate-base-result.html).

| dataset | flavor | IQM | mean ± std | L |
|---|---|--:|--:|--:|
| heart (acc ↑) | split | 0.909 | 0.909 ± 0.004 | 3 |
|  | mixed 🥇 | **0.910** | 0.909 ± 0.006 | 2 |
|  | mixed-fixed | 0.906 | 0.906 ± 0.003 | 3 |
|  | alternate | 0.906 | 0.906 ± 0.001 | 3 |
| auto (MSE ↓) | split | 10.90 | 10.88 ± 0.49 | 2 |
|  | mixed | 10.46 | 10.44 ± 0.28 | 3 |
|  | mixed-fixed 🥇 | **10.13** | 10.18 ± 0.28 | 3 |
|  | alternate | 10.14 | 10.13 ± 0.19 | 2 |
| compas (acc ↑) | split | 0.730 | 0.729 ± 0.003 | 3 |
|  | mixed | 0.704 | 0.704 ± 0.002 | 2 |
|  | mixed-fixed | 0.721 | 0.721 ± 0.002 | 4 |
|  | alternate 🥇 | **0.730** | 0.730 ± 0.002 | 3 |
| blog (RMSE ↓) | split 🥇 | **0.177** | 0.177 ± 0.002 | 2 |
|  | mixed | 0.178 | 0.178 ± 0.001 | 2 |
|  | mixed-fixed | 0.182 | 0.182 ± 0.001 | 2 |
|  | alternate | 0.186 | 0.186 ± 0.001 | 2 |
| loan (acc ↑) | split | 0.704 | 0.704 ± 0.001 | 3 |
|  | mixed | 0.704 | 0.704 ± 0.000 | 2 |
|  | mixed-fixed 🥇 | **0.708** | 0.708 ± 0.001 | 3 |
|  | alternate | 0.703 | 0.703 ± 0.000 | 2 |

**Two findings.** (1) **Fixing `convex_fraction = 0.5` beats searching it** —
`mixed-fixed` wins 3 of 5 datasets, substantially on `compas` (+0.017 acc) and
`auto` (−0.33 MSE); searching the split (the `mixed` rows) mostly hurt on the
mid/large datasets. (2) At ≤ 4 tuned layers, **`alternate` does not decisively
beat the best non-alternate flavor**: it co-leads `compas` (tied with `split`)
and sits within noise on `auto`/`heart`, but loses on `blog`/`loan` — so it
reaches parity where `mixed` was dominating without overtaking it.

Depth was studied separately: a controlled
[size-ladder study](https://davorrunje.github.io/mononet/benchmarks/loan-size-ladder.html)
finds deep/`residual` stacks are within noise of shallow ones and the edge does
not grow with scale, so **≤ 4 layers is the right regime** — which is why this
bake-off is shallow-only.

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
