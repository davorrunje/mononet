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

## Reproduction

Run on `gpu-jax` (JAX GPU). Each panel tunes every method with an identical
Optuna budget (once, seed 0), then evaluates the best config over 10 seeds and
reports the **IQM** with a 95 % bootstrap band. ~30–40 min each on a modern GPU.

```bash
# Inverse flagship (traffic state estimation) — the headline table
uv run python -m applications.pinn.experiments.headline \
    --problem lwr_riemann --tier inverse \
    --out applications/pinn/results/inverse-headline.json

# Forward mechanism tier (Burgers shock) — the contrast panel
uv run python -m applications.pinn.experiments.headline \
    --problem burgers_riemann --tier forward \
    --out applications/pinn/results/forward-mechanism.json
```

Both write per-method best params + per-seed metrics + IQM/band to the JSON.
The `§6` tables in `paper/paper.md` are filled from these artifacts.

### Still to do (need a different environment / more compute)

- [ ] **Cross-backend equivalence** (JAX vs PyTorch): needs both backends — run on
      the `default` (`all-cpu`) devcontainer; `gpu-jax` has no Torch.
- [ ] **Figures** (TV(t) curves, shock profiles, reconstructed field, sparsity ×
      noise sweep): `core/plotting.py` exists; generation + the sweep are not yet
      wired into a committed script.
- [ ] **Notebook** render into Sphinx docs.
