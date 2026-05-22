# Benchmarks

Reproducing the paper. These notebooks reproduce experiments from
[Runje & Shankaranarayana (2023)](https://arxiv.org/abs/2205.11775) using
`mononet`. They are committed with their outputs and re-executed manually
before each release — see
[`CONTRIBUTING.md`](../about/contributing.md).

Each notebook also benchmarks against `airtai/monotonic-nn`
(the paper's original PyTorch reference) installed at notebook-execution
time via `--no-deps` (see `tools/execute-benchmarks.sh`).

## Notebooks

- [Overview](00-overview.ipynb) — placeholder; full set of benchmarks
  lands in the follow-up algorithm plan.

```{toctree}
:hidden:
:maxdepth: 1

00-overview
```
