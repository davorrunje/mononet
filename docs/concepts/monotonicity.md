# Monotonicity

A function `f(x_1, ..., x_n)` is **monotonically non-decreasing** in input
`x_i` if `x_i ≤ x_i'` (holding others fixed) implies `f(...) ≤ f(...)`.
Symmetrically, it is **non-increasing** if `x_i ≤ x_i'` implies
`f(...) ≥ f(...)`. A `0` entry in the monotonicity mask means no
constraint on that input.

mononet implements the construction from Runje &
Shankaranarayana (2023) which yields **provably monotonic** networks
without constraining the underlying weights — see the paper at
<https://arxiv.org/abs/2205.11775> for the proof.

## API

See [`MonotonicityMask`](../api/mononet/core/types/MonotonicityMask.md) for
the type used to declare per-input monotonicity in code.
