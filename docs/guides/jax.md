# JAX guide

`mononet.jax` uses **Flax NNX** — the object-oriented Flax API. Layers are
`flax.nnx.Module` subclasses, fully compatible with {py:func}`jax.jit` and
{py:func}`jax.grad`, and compose with `flax.nnx.Sequential`.

## Install

    pip install "mononet[jax]"

## Public API

- {py:class}`mononet.jax.layers.MonoLinear` — monotonic analogue of
  `flax.nnx.Linear` (non-decreasing in all inputs).
- {py:class}`mononet.jax.layers.MonoResidual` — dual-gated monotone
  residual block, warm-started near identity.
- {py:class}`mononet.jax.layers.MonoInput` — sign-flip layer encoding
  per-feature monotonicity directions.

`mononet` ships layers only — stack them yourself; there is no composed
`MonoMLP` model.

## Example

A mixed-feature network: monotone in 3 features (2 non-decreasing, 1
non-increasing) via {py:class}`mononet.jax.layers.MonoInput`, and
unconstrained in 2 non-monotone features, which are embedded through a plain
MLP before being concatenated with the monotone path. The embedding absorbs
the non-monotonicity, so the composite `RiskNet` is monotone in `x_mono` and
free in `x_free`. `MonoLinear` and `MonoResidual` default to `mode="absolute"`.

```{literalinclude} ../examples/risk_net_jax.py
:language: python
```

The dense layers take an explicit `rngs` ({py:class}`flax.nnx.Rngs`) for
weight initialization. For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of
`{-1, +1}`) to `MonoInput`.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
