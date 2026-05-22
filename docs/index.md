---
hide-toc: false
---

# mononet

**Unconstrained monotonic neural networks** with first-class support for
**PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

Reference implementation of:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023. [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

## Install

```
pip install "mononet[torch]"      # PyTorch
pip install "mononet[jax]"        # JAX + Flax NNX
pip install "mononet[keras]"      # Keras 3
pip install "mononet[all]"        # all three
```

```{toctree}
:maxdepth: 1
:caption: Getting started

guides/pytorch
guides/jax
guides/keras
```

```{toctree}
:maxdepth: 1
:caption: Concepts

concepts/monotonicity
concepts/layers
```

```{toctree}
:maxdepth: 1
:caption: Benchmarks

benchmarks/index
```

```{toctree}
:maxdepth: 2
:caption: API reference

apidocs/mononet/mononet
```

```{toctree}
:maxdepth: 1
:caption: About

about/license
about/changelog
about/citation
contributing
```
