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

A mixed-feature network: monotone in 3 features (2 non-decreasing, 1
non-increasing) via {py:class}`mononet.torch.layers.MonoInput`, and
unconstrained in 2 non-monotone features, which are embedded through a plain
MLP before being concatenated with the monotone path. The embedding absorbs
the non-monotonicity, so the composite `RiskNet` is monotone in `x_mono` and
free in `x_free`. `MonoLinear` and `MonoResidual` default to `mode="absolute"`.

```{literalinclude} ../examples/risk_net_torch.py
:language: python
```

For per-feature monotonicity directions, pass a
{py:class}`~mononet.core.types.MonotonicityMask` (a 1-D array of
`{-1, +1}`) to `MonoInput`.

## See also

- [Concepts: monotonicity](../concepts/monotonicity.md)
- [Layer reference](../concepts/layers.md)
- [Benchmarks](../benchmarks/index.md)
