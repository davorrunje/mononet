# JAX guide

`mononet.jax` uses **Flax NNX** — the new object-oriented Flax API. Layers
are `flax.nnx.Module` subclasses, fully compatible with `jax.jit` and
`jax.grad`.

## Install

    pip install "mononet[jax]"

## Public API

- [`MonoLinear`](../api/mononet/jax/MonoLinear.md)
- [`MonoMLP`](../api/mononet/jax/MonoMLP.md)

A worked example lands once the algorithm implementation is in.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Benchmarks](../benchmarks/index.md)
