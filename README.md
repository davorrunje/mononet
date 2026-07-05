# mononet — Constrained Monotonic Neural Networks

[![PyPI version](https://img.shields.io/pypi/v/mononet)](https://pypi.org/project/mononet/)
[![Python versions](https://img.shields.io/pypi/pyversions/mononet)](https://pypi.org/project/mononet/)
[![Docs](https://img.shields.io/badge/docs-mononet-blue)](https://davorrunje.github.io/mononet/)
[![Build](https://github.com/davorrunje/mononet/actions/workflows/build.yml/badge.svg)](https://github.com/davorrunje/mononet/actions/workflows/build.yml)

Multi-backend implementation of the constrained monotonic neural network
construction from:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

with the activation-switch refinement (the default `mode="switch"`) from:

> Sartor, D. et al. (2025). *Advancing Constrained Monotonic Neural
> Networks.* ICML 2025. <https://arxiv.org/abs/2505.02537>

First-class support for **PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

## Quick start

`mononet` ships **layers**, not composed models — stack them with your
framework's native `Sequential` (or equivalent). Each backend exposes
`MonoResidual`, `MonoInput`, and the framework-idiomatic dense layer:
`MonoLinear` for PyTorch and JAX, `MonoDense` for Keras.

```python
import torch
from mononet.torch import MonoInput, MonoLinear

# A monotonic MLP: non-decreasing in every input feature.
net = torch.nn.Sequential(
    MonoInput(1),                    # +1 => non-decreasing; -1 => non-increasing
    MonoLinear(4, 32, mode="switch"),
    MonoLinear(32, 1, mode="switch"),
)
y = net(torch.randn(8, 4))           # (8, 1), guaranteed monotone in all inputs
```

For per-feature monotonicity directions, pass a
`mononet.core.types.MonotonicityMask` (a 1-D array of `{-1, +1}`) to
`MonoInput`. The same layers exist under `mononet.jax` and
`mononet.keras`; see the [per-backend guides](docs/guides/).

## License

Apache License 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md).
Commercial use is permitted. The technique is described in U.S. Patent
11,551,063 (assignee: AIRT Technologies Ltd.); the Apache-2.0 license
grants the patent rights needed to use this code. For academic use, please
cite the paper (see [`NOTICE.md`](NOTICE.md)).

## Formal proofs

Every theorem in the paper is mechanized in Lean 4 + mathlib4 under
[`proofs/`](proofs/). See
[the cross-reference page](https://davorrunje.github.io/mononet/concepts/proofs.html)
for the paper-claim ↔ Lean-theorem ↔ Python-test mapping.

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
