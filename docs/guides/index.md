# Guides

`mononet` ships monotonic **layers**, not composed models — stack them with your
framework's native `Sequential` (or equivalent); there is no composed `MonoMLP`.
Every backend exposes the same three layers: the **dense monotone layer** (a
monotonic analogue of the framework's dense layer, non-decreasing in all
inputs), **`MonoResidual`** (a dual-gated monotone residual block, warm-started
near identity), and **`MonoInput`** (a sign-flip layer encoding per-feature
monotonicity directions).

## Example

A mixed-feature network: monotone in 3 features (2 non-decreasing, 1
non-increasing) via `MonoInput`, and unconstrained in 2 non-monotone features,
which are embedded through a plain MLP before being concatenated with the
monotone path. The embedding absorbs the non-monotonicity, so the composite
`RiskNet` is monotone in `x_mono` and free in `x_free`. The dense layer and
`MonoResidual` default to `mode="absolute"`. Pick your backend:

::::{tab-set}
:::{tab-item} PyTorch
`mononet.torch` provides monotonic layers as {py:class}`torch.nn.Module`
subclasses; they drop into any training loop (plain PyTorch, PyTorch Lightning,
…) and compose with {py:class}`torch.nn.Sequential`.

    pip install "mononet[torch]"

Layers: {py:class}`mononet.torch.layers.MonoLinear` (monotonic
{py:class}`torch.nn.Linear`), {py:class}`mononet.torch.layers.MonoResidual`,
{py:class}`mononet.torch.layers.MonoInput`.

```{literalinclude} ../examples/risk_net_torch.py
:language: python
```

For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of `{-1, +1}`) to
`MonoInput`.
:::
:::{tab-item} JAX
`mononet.jax` uses **Flax NNX** — layers are `flax.nnx.Module` subclasses, fully
compatible with {py:func}`jax.jit` and {py:func}`jax.grad`, and compose with
`flax.nnx.Sequential`.

    pip install "mononet[jax]"

Layers: {py:class}`mononet.jax.layers.MonoLinear` (monotonic `flax.nnx.Linear`),
{py:class}`mononet.jax.layers.MonoResidual`,
{py:class}`mononet.jax.layers.MonoInput`.

```{literalinclude} ../examples/risk_net_jax.py
:language: python
```

The dense layers take an explicit `rngs` ({py:class}`flax.nnx.Rngs`) for weight
initialization. For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of `{-1, +1}`) to
`MonoInput`.
:::
:::{tab-item} Keras 3
`mononet.keras` uses `keras.ops`, so the same code runs whether Keras is
configured to use JAX, TensorFlow, or PyTorch under the hood (the GPU
devcontainer ships with `KERAS_BACKEND=jax`).

    pip install "mononet[keras]"

Layers: {py:class}`mononet.keras.layers.MonoDense` (monotonic
`keras.layers.Dense`), {py:class}`mononet.keras.layers.MonoResidual`,
{py:class}`mononet.keras.layers.MonoInput`.

```{literalinclude} ../examples/risk_net_keras.py
:language: python
```

`MonoDense` infers the input width at build time (Keras style) — no
`in_features`. `MonoDense` and `MonoInput` implement `get_config`/`from_config`,
so models serialize with the standard Keras saving APIs. For per-feature
monotonicity directions, pass a {py:class}`~mononet.core.types.MonotonicityMask`
(a 1-D array of `{-1, +1}`) to `MonoInput`.
:::
::::

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
