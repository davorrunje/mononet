# Keras 3 guide

`mononet.keras` uses `keras.ops`, so the same code runs whether Keras is
configured to use JAX, TensorFlow, or PyTorch under the hood. The
GPU devcontainer ships with `KERAS_BACKEND=jax`.

## Install

    pip install "mononet[keras]"

## Public API

- `MonoDense` — monotonic analogue of `keras.layers.Dense`.
- `MonoMLP`

A worked example lands once the algorithm implementation is in.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Benchmarks](../benchmarks/index.md)
