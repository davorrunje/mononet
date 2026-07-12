# Structure-Preserving PINNs — Paper 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-07-12-applications-structure-preserving-pinns-design.md`](../specs/2026-07-12-applications-structure-preserving-pinns-design.md). Read it first; this plan implements **Paper 1 only**.

**Goal:** Stand up the `applications/` area and the `applications/pinn/` package: the framework (admissibility abstraction, problem registry, backend-agnostic NumPy `core/`, per-backend JAX+Torch trainers, Optuna HP search), the **forward conservation-law mechanism tier** (Burgers-Riemann, Burgers smooth→shock, linear advection, LWR), and the **deep inverse flagship** (traffic state estimation from sparse noisy data). Produce a Markdown manuscript, an executed notebook, and a RUNBOOK — with the manuscript **scaffolded before any implementation**.

**Architecture:** `applications/` is repo-only, out of the `mononet` wheel. `core/` is pure NumPy (single source of truth); framework code lives only in `models/{jax,torch}` and `training/{jax,torch}_trainer.py`. Each PDE is a plug-in `core/problems/*.py` module behind one `Problem` protocol. Both tiers share the field `u_θ(x,t)` (shallow `MonoResidual`, ≈4 layers) and differ only in the loss.

**Tech Stack:** Python 3.11+, NumPy, JAX + Flax NNX + optax, PyTorch, Optuna (v4.9.0, already present), matplotlib, pytest, Sphinx/myst-nb. `mononet.jax` / `mononet.torch` layers (Sub-project A, locked API).

## Global Constraints

- **No `mononet` package/kernel/layer change.** Consume the locked public API only (`MonoLinear`/`MonoDense`, `MonoResidual`, `MonoInput`, `MonotonicityMask`, `MonoConfig`, `MonoResidualConfig`). If a genuine gap appears, STOP and raise it — do not patch the wheel from here.
- **Lazy backend imports preserved.** `import mononet` must not import torch/jax/keras. Backend code imports `from mononet.jax …` / `from mononet.torch …` locally.
- **`core/` imports no ML framework** — NumPy only. This is what makes JAX and Torch numbers comparable and is asserted by a test.
- **Monotone-solution class only** (spec §4 non-goals). Every problem module documents its scope; non-monotone data is out.
- **Determinism:** all point/observation generation is seeded in NumPy and identical across backends and methods.
- Line length 88 (ruff); **strict mypy** — `applications/` is added to the mypy `files` list and must pass. MyST field-list docstrings on all public functions/classes.
- Backends optional in CI: use `pytest.importorskip`; select with `MONONET_TEST_BACKEND`. Heavy training/search marked `slow` and excluded from default CI.
- **Branch:** `spec/applications-structure-preserving-pinns` (already checked out). Commit per task. **Never commit to `main`.** Pre-commit must pass (no `--no-verify`); locale-only errors → prefix `LC_ALL=C.UTF-8 LANG=C.UTF-8`.
- **Devcontainer strategy** (only `default` has both backends; `gpu-jax`=JAX-only, `gpu-torch`=torch-only):
  - **Phases 0–4 (dev): `default`** (CPU, `MONONET_EXTRAS=all-cpu` → both backends). Writing/unit-testing both backends and the cross-backend equivalence test require both installed in one venv; smoke/unit tests are cheap on CPU.
  - **Phases 5–6 (heavy Optuna search + sweeps): `gpu-jax` is the primary GPU backend** (JAX + `jit` for the repeated residual/Hessian). Run the full search/sweep in JAX there.
  - **Cross-backend result = equivalence test + reproducing the JAX-tuned config on Torch** (forward-equivalence + a confirmation run) — cheap, no second HP search. Do the Torch confirmation on `default` (CPU) or a short `gpu-torch` session; do **not** attempt a combined torch+jax GPU venv (CUDA 13 vs 12.9 wheel conflicts).
- Tests live under `tests/applications/pinn/` mirroring the package.

---

## Phase 0 — Area scaffold & package skeleton

### Task 0.1: `applications/` area + `pinn/` package skeleton

**Files:**
- Create: `applications/README.md`, `applications/__init__.py`, `applications/_common/{__init__,seeding,metrics_io,plot_theme}.py`
- Create: `applications/pinn/{__init__.py,README.md,RUNBOOK.md}` and package dirs `core/`, `core/problems/`, `models/{jax,torch}/`, `training/`, `configs/`, `experiments/`, `results/`, `notebooks/`, `paper/figures/`, `studies/`
- Modify: `pyproject.toml` (add `applications` to mypy `files`; add optax + flax to a dev/pinn group only if not already present — confirm before adding)
- Test: `tests/applications/pinn/test_skeleton.py`

**Interfaces:**
- `_common/seeding.py`: `rng(seed: int) -> numpy.random.Generator`, `split_seeds(seed, n) -> list[int]`.
- `_common/metrics_io.py`: `write_result(path, obj: dict)` / `read_result(path) -> dict` (JSON).

**Steps:**
- [ ] **Step 1 (RED):** `test_skeleton.py` asserts (a) `applications.pinn` importable without importing torch/jax (`sys.modules` check), (b) `seeding.rng(0)` reproducible, (c) `metrics_io` round-trips a dict.
- [ ] **Step 2 (GREEN):** create the tree + minimal module bodies with MyST docstrings.
- [ ] **Step 3:** README/RUNBOOK stubs; `applications/README.md` explains applications vs benchmarks and indexes the papers.
- [ ] **Step 4:** wire mypy/ruff; run `uv run ruff check`, `uv run mypy`, `uv run pytest tests/applications -q`.
- [ ] **Step 5:** commit `feat(pinn): applications/ area + pinn package skeleton`.

> **REVIEW CHECKPOINT 0** — confirm dependency additions are acceptable and the lazy-import test passes.

---

## Phase 1 — Markdown paper scaffold (NO implementation) — spec §11

Write everything that does not depend on results. **No code, no numbers — placeholders only.**

### Task 1.1: `references.bib`
- [ ] Gather citations from spec §9/§9a: Raissi 2019; Han–Jentzen–E 2018; Sirignano–Spiliopoulos 2018; LeVeque 2002; Runje 2023 & Sartor 2025; Shi et al. 2021 (PINN-TSE); WE-PINN, DC-PINN; HardNet 2024; ICNN (Amos 2017). Mark unverified 2026 arXiv IDs `note = {VERIFY}`.
- [ ] commit `docs(pinn-paper): references.bib`.

### Task 1.2: `paper.md` scaffold
Write in full except results (which get explicit placeholders):
- [ ] Title, author, abstract (headline number → `[[TVD-VIOLATION-NUMBER]]`).
- [ ] **1 Introduction** — soft-vs-hard constraints; the named PINN failure (Gibbs at shocks); thesis; contributions.
- [ ] **2 Related work** — grounded in §9/§9a: soft-penalty conservation/shape PINNs + PINN-TSE (Shi 2021), hard-but-inexpressive nets (Dugas/ICNN), expressiveness (HardNet, mononet).
- [ ] **3 Method** — admissibility abstraction; registry; `mononet` field, mask `x→−1`,`t→0`; shallow `MonoResidual`; loss terms; cross-backend.
- [ ] **4 Theory** — TVD/entropy-by-construction for the monotone-solution class (precise, with the class restriction); continuous-ramp shock representation.
- [ ] **5 Experimental design** — benchmarks (forward + inverse), baselines, metrics (§6), Optuna equal-budget (§6a), sparsity×noise sweep, cross-backend equivalence.
- [ ] **6 Results** — subsection headers + empty tables/figures with placeholders (`[[TABLE-forward-tier]]`, `[[FIG-inverse-sweep]]`, `[[FIG-tv-curve]]`, …), one-line captions saying what each will show.
- [ ] **7 Discussion / limitations** — monotone-class scope; forward tier is a mechanism check; pointer to Papers 2–5.
- [ ] Verify no orphan placeholder lacks a planned experiment; commit `docs(pinn-paper): manuscript scaffold with results placeholders`.

> **REVIEW CHECKPOINT 1** — human reads the scaffolded manuscript; everything except numbers reads as a finished paper. Adjust framing here before writing code.

---

## Phase 2 — `core/` backend-agnostic science (TDD, NumPy only)

### Task 2.1: `Problem` protocol + registry
**Files:** `core/problems/base.py`, `core/problems/__init__.py`; Test `test_registry.py`
- [ ] RED: registry register/get/available; `get("nope")` raises. GREEN + commit.

### Task 2.2: `admissibility.py`
**Files:** `core/admissibility.py`; Test `test_admissibility.py`
- [ ] RED: monotone-decreasing field → violation 0; a bump → violation = bump rise. GREEN + commit.

### Task 2.3: `exact.py` — closed-form entropy solutions
**Files:** `core/exact.py`; Test `test_exact.py`
- Interfaces: `burgers_riemann(x,t,uL,uR)` (shock, `s=(uL+uR)/2`); `burgers_smooth_shock(x,t,u0)` (characteristics pre-`t_b`, R-H post; `t_b=-1/min(u0')`); `advection(x,t,a,u0)`; `lwr_riemann(x,t,ρL,ρR,flux)`.
- [ ] RED: Rankine–Hugoniot speed, self-similarity, monotonicity preserved, sampled values. GREEN + commit.

### Task 2.4: `reference_solver.py` — TVD finite-volume (Godunov)
**Files:** `core/reference_solver.py`; Test `test_reference_solver.py`
- [ ] RED: converges to `burgers_riemann` under refinement (L¹↓ below tol); TV non-increasing; correct shock speed. GREEN + commit.

### Task 2.5: `sampling.py` — deterministic point sets + sparse observations
**Files:** `core/sampling.py`; Test `test_sampling.py`
- Interfaces: `collocation`, `initial_points`, `boundary_points`, `eval_grid`, `observations(reference_field, grid, n_obs, noise_std, seed)`.
- [ ] RED: same seed → identical; obs count exact; noise reproducible; coords in domain. GREEN + commit.

### Task 2.6: `conservation.py` problems (forward + inverse mode)
**Files:** `core/problems/conservation.py`; Test `test_conservation.py`
- `Burgers`, `LinearAdvection`, `LWR` (Greenshields `Q(ρ)=v_max ρ(1−ρ/ρ_max)`; document choice).
- [ ] RED: residual of exact solution ≈ 0 on interior (finite-diff); mask matches known monotone direction. GREEN + commit.

### Task 2.7: `metrics.py` + `plotting.py`
**Files:** `core/metrics.py`, `core/plotting.py`; Test `test_metrics.py`
- `l1,l2,tv,tv_curve,overshoot,shock_position_error,mass_error,reconstruction_error`; plotting returns Figures (no I/O).
- [ ] RED: metrics on analytic inputs (monotone TV = endpoint diff; zero overshoot). GREEN + commit.

> **REVIEW CHECKPOINT 2** — `core/` complete, framework-free (assert no torch/jax import), validated against closed forms. This is the ruler.

---

## Phase 3 — Models & trainers (per-backend, TDD)

### Task 3.1: model protocol + builders (both backends)
**Files:** `models/protocol.py`, `models/jax/*.py`, `models/torch/*.py`; Tests `test_builders_{jax,torch}.py` (`importorskip`)
- `build(problem, cfg, method)` for `method ∈ {vanilla, soft, weight_clip, hard_monotone}`; hard = shallow `MonoResidual` (≈4 layers) with the mask; baselines matched in depth/width.
- [ ] RED: forward-pass shape; `hard_monotone` monotone in `x` on a grid (finite-diff sign) per backend. GREEN + commit each.

### Task 3.2: cross-backend forward-equivalence test
**Files:** `test_cross_backend.py`
- [ ] With identical ported weights, JAX vs Torch `hard_monotone` agree within tol; skip unless both importable. commit.

### Task 3.3: losses + trainers
**Files:** `training/losses.py`, `training/jax_trainer.py`, `training/torch_trainer.py`; Tests `test_trainer_{jax,torch}.py`
- `losses.py` term specs (residual, IC, BC, data-fit). Trainers: `train(problem, method, cfg, points, backend)`; residual via `jax.grad/hessian` / `autograd.grad(create_graph=True)`.
- [ ] RED (smoke): few-step train on Burgers-Riemann reduces loss; `hard_monotone` violation 0 throughout. Full runs `slow`. GREEN + commit.

> **REVIEW CHECKPOINT 3** — smoke train runs on both backends; monotone model provably non-oscillatory.

---

## Phase 4 — Experiments, Optuna search, configs

### Task 4.1: `experiments/run.py` (Typer CLI); Test `test_run_smoke.py`
- [ ] `run(problem, method, backend, seed, config)` → results JSON. RED: smoke run → valid artifact. GREEN + commit.

### Task 4.2: `experiments/search.py` — Optuna, equal budget; Test `test_search_smoke.py`
- [ ] Reuse `benchmarks/_common/search.py` pattern. **Identical search space + `n_trials` across all methods**; soft penalty weight IS searched. Freeze best config per (problem, method, backend) to `configs/`. RED: 2-trial smoke → frozen config. GREEN + commit.

### Task 4.3: `experiments/sweep.py` + configs; Test `test_sweep_smoke.py`
- [ ] Enumerate matrix (problem × method × backend × seed × obs-sparsity/noise) consuming tuned configs. RED: smoke 2-cell sweep. GREEN + commit.

> **REVIEW CHECKPOINT 4** — machinery works at smoke scale; RUNBOOK updated with search+sweep commands.

---

## Phase 5 — Forward mechanism tier (results) — `gpu-jax`

### Task 5.1: run forward tier + references
- [ ] `slow`: Optuna search (equal budget) + multi-seed sweep for all four forward problems, all methods, both backends where feasible. Cache TVD references to `results/<problem>/reference.npz` (committed).
- [ ] Assert headline: `hard_monotone` violation = 0, overshoot ≈ 0; baselines show Gibbs. commit results.

---

## Phase 6 — Inverse flagship (traffic state estimation) (results) — `gpu-jax`

### Task 6.1: inverse problem wiring
- [ ] Define monotone-front scenarios (queue behind bottleneck / red signal / incident — resolve spec §10; document). Observation model. RED: inverse loss uses observations, no full IC. GREEN + commit.

### Task 6.2: sparsity × noise sweep + results
- [ ] `slow`: reconstruct from sparse noisy observations across the sparsity×noise grid, all methods, both backends. Reference = high-res Godunov field.
- [ ] Headline curves: reconstruction L¹/L² vs sparsity/noise (hard-monotone holds; soft/vanilla degrade+oscillate); violation → 0; front-position error; cross-backend equivalence. commit results.

> **REVIEW CHECKPOINT 6** — the headline result exists and matches the scaffolded claims (or the claims get revised to match reality — evidence over assertion).

---

## Phase 7 — Fill manuscript, notebook, docs

### Task 7.1: fill results into `paper.md`
- [ ] Replace every `[[…]]` placeholder with real tables/numbers/figures. If a result contradicts a scaffolded claim, revise the claim. commit.

### Task 7.2: notebook + Sphinx integration
- [ ] Executed notebook (loads committed results/references — no heavy training at build time). Add "Applications" to the Sphinx toctree. RED: `./tools/build-docs.sh` renders it. GREEN + commit.

### Task 7.3: finalize RUNBOOK + README headline
- [ ] RUNBOOK reproduces every figure/table (search → sweep → fill → notebook). `pinn/README.md` gets abstract + headline number. commit.

> **REVIEW CHECKPOINT 7 (final)** — full suite green (excluding `slow` in CI), `ruff`/`mypy` clean, docs build, manuscript complete. Then use superpowers:finishing-a-development-branch.

---

## Out of scope (registry-ready, not implemented here)
- HJB (Paper 2, with §9a caveats), Fokker–Planck (Paper 3), eikonal (Paper 4): only the registry must admit them.
- Arbitrage-free surfaces (Paper 5): separate, non-PINN, own brainstorm.
- Adaptive resampling (RAR): optional `slow` ablation only, not the headline.
