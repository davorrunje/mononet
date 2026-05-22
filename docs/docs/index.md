---
hide:
  - navigation
search:
  exclude: false
---

# mononet

**Unconstrained monotonic neural networks**, with first-class support for
**PyTorch**, **JAX** (Flax NNX), and **Keras 3**.

Reference implementation of:

> Runje, D., Shankaranarayana, S. M. (2023). *Constrained Monotonic
> Neural Networks.* ICML 2023.
> [arXiv:2205.11775](https://arxiv.org/abs/2205.11775)

## Install

    pip install "mononet[torch]"      # PyTorch
    pip install "mononet[jax]"        # JAX + Flax NNX
    pip install "mononet[keras]"      # Keras 3
    pip install "mononet[all]"        # all three

## Where to go next

- [PyTorch guide](guides/pytorch.md)
- [JAX guide](guides/jax.md)
- [Keras guide](guides/keras.md)
- [Concepts: monotonicity](concepts/monotonicity.md)
- [Benchmarks (reproducing the paper)](benchmarks/index.md)

## License & patent

Code: PolyForm Noncommercial 1.0.0. Patent: US 11,551,063 reserved
(assignee: AIRT Technologies Ltd.). Commercial users contact
**licensing@airt.ai**. See [License & patent](about/license.md).
