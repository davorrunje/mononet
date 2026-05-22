# Layer reference

Each backend mirrors its host framework's vocabulary for the analogous
unconstrained layer:

| Concept                  | PyTorch                  | JAX (Flax NNX)             | Keras 3                       |
|--------------------------|--------------------------|----------------------------|-------------------------------|
| Single monotonic layer   | `MonoLinear`             | `MonoLinear`               | `MonoDense`                   |
| Composed MLP             | `MonoMLP`                | `MonoMLP`                  | `MonoMLP`                     |

PyTorch and Flax NNX both call the standard analog `Linear`, so the
monotonic version is `MonoLinear` in those backends. Keras calls it
`Dense`, so the monotonic version is `MonoDense`.

The composed MLP shares the name `MonoMLP` across all three backends
since "MLP" is universal.

Pure-function NumPy reference implementations under
`mononet.core.reference` (`monotonic_dense`, `monotonic_mlp`) provide the
arithmetic ground truth used by the cross-backend equivalence tests.
