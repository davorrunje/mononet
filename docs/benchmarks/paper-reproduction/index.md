# Reproducing the paper

These notebooks reproduce the benchmark results from
[Runje & Shankaranarayana (2023)](https://arxiv.org/abs/2205.11775) (*Constrained Monotonic
Neural Networks*, ICML 2023) using the `mononet` package. The original datasets are
archived on Zenodo at <https://zenodo.org/records/7968969>; fetch them with:

```bash
python -m benchmarks.datasets.download
```

Each dataset notebook runs all four flavors (torch/switch, torch/absolute, jax/switch,
keras/switch) for a small epoch and seed budget so the scaffold executes quickly.
The maintainer re-runs with full budgets via `tools/execute-benchmarks.sh` and commits
the outputs before each release. The `tables` notebook aggregates the committed
`benchmarks/results/paper-reproduction.json` into the headline and cross-backend tables.

```{toctree}
:maxdepth: 1

auto-mpg
heart-disease
compas
blog-feedback
loan-defaulter
tables
```
