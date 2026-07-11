# Loan size-ladder — does depth win with scale?

PR #72 found deep monotone residual stacks beat shallow ones **only** on
`loan`, the largest dataset. This experiment isolates the cause: it holds
`loan` fixed and varies the training-set size N, tuning a **deep** (`absolute`
residual, depth ∈ {6, 10, 16}) and a **shallow** (`absolute` residual, depth ∈
[1, 4]) arm independently at each N, then reports

$$\Delta(N) = \mathrm{IQM}_{\text{deep}}(N) - \mathrm{IQM}_{\text{shallow}}(N)$$

on the full held-out test set (10 seeds per arm; a fresh stratified
N-subsample per seed, so the IQM band captures subsample and training
variance). Method and protocol: {doc}`protocol` and the
[design spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md).

```{note}
Results are produced by the GPU session per `benchmarks/RUNBOOK-loan-ladder.md`;
this page is populated (plot + table) when that run lands.
```

<!-- The GPU run replaces this block with the Δ-vs-N plot and per-N table. -->
