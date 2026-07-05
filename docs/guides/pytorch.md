# PyTorch guide

`mononet.torch` provides monotonic layers as {py:class}`torch.nn.Module`
subclasses. They drop into any existing training loop (plain PyTorch,
PyTorch Lightning, etc.) and compose with the native
{py:class}`torch.nn.Sequential`.

## Install

    pip install "mononet[torch]"

## Public API

- {py:class}`mononet.torch.layers.MonoLinear` — monotonic analogue of
  {py:class}`torch.nn.Linear` (non-decreasing in all inputs).
- {py:class}`mononet.torch.layers.MonoResidual` — dual-gated monotone
  residual block, warm-started near identity.
- {py:class}`mononet.torch.layers.MonoInput` — sign-flip layer encoding
  per-feature monotonicity directions.

`mononet` ships layers only — stack them yourself; there is no composed
`MonoMLP` model.

## Example

```python
import torch
from mononet.torch import MonoInput, MonoLinear

# A monotonic MLP: non-decreasing in every input feature.
net = torch.nn.Sequential(
    MonoInput(1),                     # +1 => non-decreasing; -1 => non-increasing
    MonoLinear(4, 32, mode="switch"),
    MonoLinear(32, 1, mode="switch"),
)
y = net(torch.randn(8, 4))            # (8, 1), guaranteed monotone in all inputs
```

For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of
`{-1, +1}`) to `MonoInput`.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
