# RUNBOOK — Structure-Preserving PINNs (Paper 1)

Exact commands to regenerate every figure and table. Filled in as the
implementation lands (see the plan:
`docs/superpowers/plans/2026-07-12-applications-structure-preserving-pinns-paper1.md`).

## Environment

Application deps are opt-in uv groups (`pinn` includes shared `applications`);
backends are package extras. Sync the app env before running anything here:

```bash
uv sync --extra jax --group pinn        # CPU JAX + optax + matplotlib (dev/tests)
uv sync --extra jax-gpu --group pinn     # GPU JAX (heavy search); needs working CUDA
uv sync --extra all-cpu --group pinn     # both backends (cross-backend equivalence)

uv run pytest tests/applications -q      # unit + smoke suite
```

- **Development / unit + smoke tests:** `default` devcontainer (`all-cpu`, both
  backends) or a single-backend env (the other backend's tests skip).
- **Heavy Optuna search + sweeps:** `gpu-jax` devcontainer (primary GPU backend).
- **Cross-backend equivalence:** needs both backends → an `all-cpu` sync.

## Reproduction (to be completed)

- [ ] Optuna search (equal budget per method) — `experiments/search.py`
- [ ] Forward mechanism-tier sweep — `experiments/sweep.py`
- [ ] Inverse flagship sparsity × noise sweep — `experiments/sweep.py`
- [ ] Fill manuscript results + render notebook
