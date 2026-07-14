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

Default field = `MonoResidual` (canonical filenames). Add `--no-residual` for the
plain-`MonoLinear` field ablation (`-plain` suffix). Both are first-class.

```bash
# Inverse flagship (traffic state estimation) — the headline table
uv run python -m applications.pinn.experiments.headline \
    --problem lwr_riemann --tier inverse \
    --out applications/pinn/results/inverse-headline.json

# Forward mechanism tier (Burgers shock) — the contrast panel
uv run python -m applications.pinn.experiments.headline \
    --problem burgers_riemann --tier forward \
    --out applications/pinn/results/forward-mechanism.json

# Plain-MonoLinear field ablation (both tiers)
uv run python -m applications.pinn.experiments.headline --no-residual \
    --problem lwr_riemann --tier inverse \
    --out applications/pinn/results/inverse-headline-plain.json
uv run python -m applications.pinn.experiments.headline --no-residual \
    --problem burgers_riemann --tier forward \
    --out applications/pinn/results/forward-mechanism-plain.json

# Sparsity x noise sweep (residual + plain). Re-running with an extended noise
# grid resumes: cells already in --out are kept, only new ones computed.
uv run python -m applications.pinn.experiments.sweep_inverse \
    --tuned applications/pinn/results/inverse-headline.json \
    --out applications/pinn/results/inverse-sweep.json
uv run python -m applications.pinn.experiments.sweep_inverse --no-residual \
    --tuned applications/pinn/results/inverse-headline-plain.json \
    --out applications/pinn/results/inverse-sweep-plain.json

# Non-PINN smoothness comparator (classical RBF smoother; CPU-only, no PDE)
uv run python -m applications.pinn.experiments.baselines \
    --out applications/pinn/results/inverse-baseline-smoother.json

# Paper figures: crossover curves (from the sweep JSONs) + reconstruction slices
# (retrains the six tuned inverse configs at the stress point). Writes PNGs to
# paper/figures/. Add --no-reconstructions to rebuild only the curves (no training).
uv run python -m applications.pinn.experiments.figures
```

Each writes per-method best params + per-seed metrics + IQM/band to the JSON;
the JSON records `"field": "residual"|"plain"`. The `§6` tables in `paper/paper.md`
are filled from these artifacts.

### Still to do (need a different environment / more compute)

- [ ] **Cross-backend equivalence** (JAX vs PyTorch): needs both backends — run on
      the `default` (`all-cpu`) devcontainer; `gpu-jax` has no Torch.
- [x] **Figures** — reconstruction slices + L¹/L²/admissibility crossover curves
      wired into `experiments/figures.py` (writes `paper/figures/*.png`). TV(t)
      curves remain a forward-tier nicety, not yet generated.
- [x] **Real-data figures** — reconstruction/window-heatmap/metric-bars for the
      NGSIM window wired into `experiments/figures.py --real` (see below).
- [ ] **Notebook** render into Sphinx docs.

## Real-data validation (NGSIM I-80)

Validates the hard-monotone field against real freeway data, not just the
synthetic Riemann problems above. Runs on CPU; no GPU needed.

```bash
# 0. Obtain raw NGSIM I-80 trajectories -> applications/pinn/data/raw/i80.csv
#    (manual; canonical hosts are unreachable from the dev container). Gitignored.

# 1. Build the derived dataset (Edie field + window scan + FD) -> LFS .npz
uv run python -m applications.pinn.data.ngsim \
    --raw applications/pinn/data/raw/i80.csv \
    --out applications/pinn/data/ngsim-i80-wave.npz
```

**Gate:** after step 1, inspect the printed `monotonicity_defect` and the
`window-ngsim` heatmap. If defect >= 0.05 or the window is too small for a
detector split, **do not proceed** to step 2 — re-run with a different
`--lane`/window, or pivot to wave-following coordinates (spec §1a).

```bash
# 2. Inverse detector-mode headline on the real field (tune-once + multi-seed)
uv run python -m applications.pinn.experiments.headline \
    --problem ngsim_wave --tier inverse --observations detectors \
    --out applications/pinn/results/real-ngsim.json

# 3. Figures (reconstruction, window heatmap, metric bars; PDF+PNG)
uv run python -m applications.pinn.experiments.figures --real
```

`figures.py --real` computes the ASM + RBF-smoother baselines inline (from the
same detector observations) — no separate `baselines.py` CLI run is needed for
the real-data panel.

`results/real-ngsim.json` does not record the detector counts used to produce
it (`run_panel`'s output only carries tuned per-method params). Step 2's
`--n-detectors`/`--n-holdout-detectors` must therefore match the constants
`figures.py --real` assumes when it retrains for reconstruction — currently
`n_detectors=8`, `n_holdout_detectors=4` (both also `headline.py`'s defaults,
so the commands above already agree; only a concern if you override either
flag).
