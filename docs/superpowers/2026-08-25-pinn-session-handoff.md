# PINN applications — session handoff / resume guide

**Purpose:** let a fresh session resume this work after a break, with no prior
memory. Read this top-to-bottom first, then the linked specs/plans/ledger.

## TL;DR — where we are

- **Paper 1 (synthetic) is done and strong:** structure-preserving PINNs for
  scalar conservation laws. Two PDEs (LWR concave, Burgers convex), honest L1/L2
  (L1 = general win under noise; L2 = front-weighted, PDE-dependent), guaranteed
  admissibility, ASM/RBF baselines, PDF+PNG figures. All committed on the work
  branch.
- **Real-data validation pivoted to sedimentation.** NGSIM traffic was explored
  and is a **dead end** for the monotone prior (freeway = multi-wave / non-
  monotone at any wave speed; arterial = non-1-D geometry). The genuinely-
  suited real dataset is **De Clercq batch-settling solids-concentration
  profiles C(z,t)** (Kynch conservation law, monotone by physics).
- **`sediment_batch` Kynch problem is built, tested, and de-risked** end-to-end on
  a synthetic field — ready for the real data.
- **Data request is out and warmly received.** Prof. Peter Vanrolleghem
  (modelEAU, co-author) is sending the **three Deinze `.mat`** datasets (and will
  look for Destelbergen); open to co-authorship if he contributes. **Currently
  awaiting the files** (a reminder was sent 2026-08-25).
- **Blocking wrinkle:** a **git history rewrite** on `main` and the PR branch left
  the complete work on a now-disjoint local line. A careful **resync** is planned
  (Option A below) but **not yet done** — it needs approval for its one
  destructive step. Everything is backed up; nothing is lost.

## What's done (on the work branch `2b22d0b` / this checkpoint)

- `applications/pinn/` — full PINN application: core (problems, exact solutions,
  Godunov reference solver, sampling, metrics, plotting, admissibility), models
  (jax/torch builders + protocol), training (jax/torch trainers), experiments
  (run, headline, search, sweep_inverse, baselines incl. ASM, figures).
- Synthetic results + figures committed under `applications/pinn/results/` and
  `applications/pinn/paper/figures/` (PDF+PNG via Git LFS for `*.pdf`).
- Manuscript draft `applications/pinn/paper/paper.md` (abstract + §6 results incl.
  the honest L1/L2 story and §6.5 real-data scaffolding).
- **Sedimentation readiness:**
  - `mononet` flux: `applications/pinn/core/exact.py::hindered_settling_flux` /
    `_prime` — Kynch/Michaels-Bolger `f_bk(c)=v0 c (1-c/c_max)^n`, backend-poly.
  - `applications/pinn/core/problems/sedimentation.py::SedimentBatch` (registered
    `"sediment_batch"`): loads a measured `C(z,t)` `.npz` (lazy scipy), monotone-
    in-height admissibility, Kynch flux; mirrors `ngsim_wave` so the detector/
    inverse pipeline reuses unchanged. De-risked: plumbing works, hard-monotone
    reconstructs correctly, admissibility 0 by construction.
- `applications/pinn/data/DATA_REQUEST.md` — the data-request email (sent).

## Real-data status & how to finish it (when Peter's data arrives)

1. Peter is sending `.mat` files (3 Deinze sets; Destelbergen to follow). They
   load with `scipy.io.loadmat`. **Paste the variable names/shapes** to the
   session and it will wire the loader.
2. Write a small loader: `.mat` C(z,t) table -> `applications/pinn/data/
   declercq-batch.npz` with keys `x`(=z height), `t`, `rho`(=C), `v0`, `c_max`,
   `n`, `sign_x`. Calibrate `v0`/`c_max`/`n` from the settling dynamics (thesis
   Table 5.2 / Kynch flux fit) so the PDE residual is **consistent** with the
   data (a de-risk finding: an inconsistent flux makes the residual fight the
   data).
3. Run the inverse detector pipeline:
   ```
   uv run python -m applications.pinn.experiments.headline \
       --problem sediment_batch --tier inverse --observations detectors \
       --out applications/pinn/results/real-sediment.json
   ```
4. **Generalize `figures.py --real`** (currently hardcoded to `ngsim_wave`) to
   accept `sediment_batch`, then regenerate figures.
5. Write manuscript §6.5 with the numbers + honest caveats. Cite De Clercq et al.
   (2005, *Water Research*) and the 2006 PhD thesis; cite the Zenodo DOI if/when
   published. Acknowledge / co-author Vanrolleghem per his contribution.

Reference: `docs/superpowers/specs/2026-07-13-real-traffic-data-ngsim-design.md`
(the pipeline design; NGSIM-specific but the inverse/detector machinery and
metrics carry over to sedimentation). The NGSIM negative finding is documented in
`.superpowers/sdd/progress.md`.

## GIT SITUATION (read before any resync) — nothing is lost

A history rewrite was done on `main` (renamed mono modes + added bench flavor-
ablation work + `mononet.legacy`), and a **subsequent** rewrite on the PR branch
attempted to reconcile with `main`. Result:

- **Complete work** lives on local branch `2b22d0b` (this checkpoint) — based on
  the **pre-rewrite** line, so **disjoint** (no merge-base) from the rewritten
  `origin/main` (`423e97e`) and `origin/spec` (`aff7f94`).
- The rewritten `origin/spec` has clean **early-phase** PINN history but is
  **missing my session's application work** (figures, `sediment_batch`, ngsim,
  results, paper updates, fixes) and is **not** reconciled with the newest `main`
  (still old mode names).
- **Backups (nothing can be lost):** local branch `backup/pinn-session-2b22d0b`,
  tag `backup-pinn-2b22d0b`, the reflog, GitHub `refs/pull/98/head` (= `91136e1`,
  everything up to the 2nd-latest commit), and this checkpoint branch/PR.

### Mode-API migration required
New `main` renamed `mononet.core.config.Mode`: **`absolute`->`mixed`,
`switch`->`split`** (plus a new `alternate`), and it **rejects** the old names
(raises). The PINN code uses `mode="absolute"` throughout, so basing on new `main`
requires migrating `absolute->mixed`, `switch->split` across `applications/pinn`.
(A stopgap CI fix already defines `Mode` locally in
`applications/pinn/models/protocol.py`; the migration supersedes it.)

### Resync strategy (Option A — approved approach; destructive step still needs OK)
1. Scratch branch `resync` from **`origin/spec`** (shares history with `main`).
2. Merge `origin/main` -> brings the rename + bench + legacy; resolve `mononet/`
   conflicts by taking `main`.
3. **Surgically** add my session's application work only (NOT the whole stale
   local tree, which would revert the rewrite's changes to
   `mononet/benchmarks/tests/docs/proofs`). Source of truth for my session work =
   this checkpoint branch, `applications/pinn/**` (plus its tests + DATA_REQUEST +
   RUNBOOK).
4. Migrate the mode API (`absolute->mixed`, `switch->split`).
5. Verify: `ruff` + `mypy` + full test suite green on the scratch branch.
6. Push to a NEW branch; confirm CI green.
7. **DESTRUCTIVE (needs explicit approval):** force-update the PR branch
   `spec/applications-structure-preserving-pinns` to the verified resync.

## Environment / tooling notes

- `uv` project; sync app env: `uv sync --extra jax --group pinn --group bench`.
- Tests: `MONONET_TEST_BACKEND=jax uv run --no-sync pytest tests/applications -q --no-cov`.
- Prefix shell with `LC_ALL=C.UTF-8 LANG=C.UTF-8`; ruff line length 88; strict mypy;
  commit on a branch, never `main`; end commit messages with the Co-Authored-By
  trailer.
- `core/*` must stay framework-free (numpy/scipy only). `flux`/`flux_prime`
  backend-polymorphic.
- PDF reading is set up (poppler + `pypdf` + PyMuPDF) — used to read the De Clercq
  thesis (in `applications/pinn/data/raw/`, gitignored).
- Container: `gpu-jax`. NGSIM raw data was cleaned up (dead end).

## Key pointers

- Specs: `docs/superpowers/specs/2026-07-12-applications-structure-preserving-pinns-design.md`,
  `docs/superpowers/specs/2026-07-13-real-traffic-data-ngsim-design.md`.
- Plans: `docs/superpowers/plans/2026-07-12-...paper1.md`,
  `docs/superpowers/plans/2026-07-13-real-traffic-data-ngsim.md`.
- Session ledger (decisions, NGSIM negative findings, de-risk): `.superpowers/sdd/progress.md`.
- Data request email: `applications/pinn/data/DATA_REQUEST.md`.
- Original PR: #98 (stale head `91136e1` after the rewrite).
