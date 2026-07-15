# Benchmarks

These notebooks reproduce experiments from
[Runje & Shankaranarayana (2023)](https://arxiv.org/abs/2205.11775) using
`mononet`. They are committed with their outputs and re-executed manually
before each release — see
[`CONTRIBUTING.md`](../about/contributing.md).

Each notebook also benchmarks against `airtai/monotonic-nn`
(the paper's original PyTorch reference) installed at notebook-execution
time via `--no-deps` (see `tools/execute-benchmarks.sh`).

## Sections

- [Protocol](protocol.md) — how we train, select, and report; and why our numbers
  differ from the original papers.
- [Overview](00-overview.ipynb) — high-level summary.
- [Reproducing the paper](paper-reproduction/index.md) — per-dataset notebooks and
  summary tables for all five benchmark datasets from the ICML 2023 paper.
- [Flavor comparison](flavor-comparison.ipynb) — Phase 2a Optuna HP-search results
  comparing the four `mode × residual` flavors (`mixed`/`split` × `plain`/`residual`).
- [Deep residual accuracy](deep-residual-accuracy.md) — does the now-trainable
  depth (residual `sub_depth=2` skips) improve real-dataset test accuracy over
  the shallow tuned flavors?
- [Loan size-ladder](loan-size-ladder.md) — does deep monotone residual win
  once the dataset is large enough?
- [Large-dataset screen](large-dataset-screen.md) — max-size deep vs shallow on
  the benchmark roster; gates each dataset to a full ladder study or the
  standard benchmark by the Δ criterion.
- [Deep-network init](deep-init.ipynb) — the static `mixed` init that fixes moderate-depth
  trainability, and the deep-depth limitation it does not solve.
- [Alternate base result](alternate-base-result.md) — tuned shallow (≤4 layers)
  `alternate` flavor vs. the best of `split`/`mixed`, on the five paper datasets.

```{toctree}
:hidden:
:maxdepth: 2

protocol
00-overview
paper-reproduction/index
flavor-comparison
deep-residual-accuracy
loan-size-ladder
large-dataset-screen
deep-init
alternate-base-result
```
