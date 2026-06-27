# mononet — Unconstrained Monotonic Neural Networks

[![PyPI version](https://img.shields.io/pypi/v/mononet)](https://pypi.org/project/mononet/)
[![Python versions](https://img.shields.io/pypi/pyversions/mononet)](https://pypi.org/project/mononet/)
[![Docs](https://img.shields.io/badge/docs-mononet-blue)](https://davorrunje.github.io/mononet/)
[![Build](https://github.com/davorrunje/mononet/actions/workflows/build.yml/badge.svg)](https://github.com/davorrunje/mononet/actions/workflows/build.yml)

Reference implementation of the unconstrained monotonic neural network
construction from:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. <https://arxiv.org/abs/2205.11775>

First-class support for **PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

## Quick start

A 60-second tour will appear here once the algorithm implementation lands.
Each backend exposes the same composed model (`MonoMLP`) and the
framework-idiomatic layer name (`MonoLinear` for PyTorch and JAX,
`MonoDense` for Keras).

```python
# PyTorch
from mononet.torch import MonoMLP

# JAX
from mononet.jax import MonoMLP

# Keras 3
from mononet.keras import MonoMLP
```

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
and benchmarks lives in [`docs/docs/`](docs/docs/).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development workflow:
devcontainer choice, `uv sync`, pre-commit, per-backend test commands.

## Citation

If you use `mononet` in academic work, please cite the paper. BibTeX is
in [`docs/docs/about/citation.md`](docs/docs/about/citation.md).
