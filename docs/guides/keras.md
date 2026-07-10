# Keras 3 guide

`mononet.keras` uses `keras.ops`, so the same code runs whether Keras is
configured to use JAX, TensorFlow, or PyTorch under the hood. The GPU
devcontainer ships with `KERAS_BACKEND=jax`.

## Install

    pip install "mononet[keras]"

## Public API

- {py:class}`mononet.keras.layers.MonoDense` — monotonic analogue of
  `keras.layers.Dense` (non-decreasing in all inputs).
- {py:class}`mononet.keras.layers.MonoResidual` — dual-gated monotone
  residual block, warm-started near identity.
- {py:class}`mononet.keras.layers.MonoInput` — sign-flip layer encoding
  per-feature monotonicity directions.

`mononet` ships layers only — stack them yourself; there is no composed
`MonoMLP` model.

## Example

A mixed-feature network: monotone in 3 features (2 non-decreasing, 1
non-increasing) via {py:class}`mononet.keras.layers.MonoInput`, and
unconstrained in 2 non-monotone features, which are embedded through a plain
MLP before being concatenated with the monotone path. The embedding absorbs
the non-monotonicity, so the composite `RiskNet` is monotone in `x_mono` and
free in `x_free`. `MonoDense` and `MonoResidual` default to `mode="absolute"`.
`MonoDense` infers the input width at build time (Keras style) — no
`in_features`.

```{literalinclude} ../examples/risk_net_keras.py
:language: python
```

For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of
`{-1, +1}`) to `MonoInput`. `MonoDense` and `MonoInput` implement
`get_config`/`from_config`, so models serialize with the standard Keras
saving APIs.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
