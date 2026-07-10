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

```python
import jax.numpy as jnp
from flax import nnx
from mononet.jax import MonoInput, MonoLinear

# A monotonic MLP: non-decreasing in every input feature.
net = nnx.Sequential(
    MonoInput(1),                                    # +1 => up; -1 => down
    MonoLinear(4, 32, mode="switch", activation="relu", rngs=nnx.Rngs(0)),
    MonoLinear(32, 1, mode="switch", rngs=nnx.Rngs(1)),  # linear read-out
)
y = net(jnp.zeros((8, 4)))                           # (8, 1), monotone in all inputs
```

The dense layers take an explicit `rngs` ({py:class}`flax.nnx.Rngs`) for
weight initialization. For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of
`{-1, +1}`) to `MonoInput`.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
