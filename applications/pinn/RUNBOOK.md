# RUNBOOK — Structure-Preserving PINNs (Paper 1)

Exact commands to regenerate every figure and table. Filled in as the
implementation lands (see the plan:
`docs/superpowers/plans/2026-07-12-applications-structure-preserving-pinns-paper1.md`).

## Environment

- **Development / unit + smoke tests:** `default` devcontainer (CPU, all backends).
- **Heavy Optuna search + sweeps:** `gpu-jax` devcontainer (primary GPU backend).

```bash
# unit + smoke suite (fast)
uv run pytest tests/applications -q
```

## Reproduction (to be completed)

- [ ] Optuna search (equal budget per method) — `experiments/search.py`
- [ ] Forward mechanism-tier sweep — `experiments/sweep.py`
- [ ] Inverse flagship sparsity × noise sweep — `experiments/sweep.py`
- [ ] Fill manuscript results + render notebook
