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

- [Overview](00-overview.ipynb) — high-level summary.
- [Reproducing the paper](paper-reproduction/index.md) — per-dataset notebooks and
  summary tables for all five benchmark datasets from the ICML 2023 paper.
- [Flavor comparison](flavor-comparison.ipynb) — Phase 2a Optuna HP-search results
  comparing the four `mode × residual` flavors (`absolute`/`switch` × `plain`/`residual`).

```{toctree}
:hidden:
:maxdepth: 2

00-overview
paper-reproduction/index
flavor-comparison
```
