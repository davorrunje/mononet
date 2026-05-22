# PyTorch guide

`mononet.torch` exposes `MonoLinear` and `MonoMLP`, both subclasses of
`torch.nn.Module`. They drop into any existing training loop (plain
PyTorch, PyTorch Lightning, etc.).

## Install

    pip install "mononet[torch]"

## Public API

- [`MonoLinear`](../api/mononet/torch/MonoLinear.md) — monotonic
  analogue of `torch.nn.Linear`.
- [`MonoMLP`](../api/mononet/torch/MonoMLP.md) — multi-layer composition.

A worked example lands once the algorithm implementation is in.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
