# Real-traffic-data validation (NGSIM I-80) — design

**Date:** 2026-07-13
**Status:** Design (approved for planning)
**Parent:** [Applications: Structure-Preserving PINNs](2026-07-12-applications-structure-preserving-pinns-design.md) — this is a Paper 1 addition (new §6.5).

## Context

Paper 1's central result — expressive hard monotonicity gives lower whole-field
(L¹) reconstruction error under noise and *guaranteed* admissibility — currently
rests entirely on **synthetic** Riemann problems with **synthetic i.i.d. Gaussian
observation noise** (LWR, Burgers). The obvious reviewer objection is that the win
is an artifact of clean synthetic dynamics + idealized noise. This deliverable
attacks that caveat directly: reproduce the inverse-reconstruction comparison on a
**real** traffic field (NGSIM I-80), with real dynamics, real aggregation noise,
and genuine LWR model-mismatch, using the *same* metrics so the result slots
beside the synthetic ones.

Scope is deliberately narrow: 1-D, inverse tier only, one field, one segment. It
is a validation section, not a new capability. Higher-D / operational PeMS-style
sparse estimation remain follow-up work.

## Constraints & decisions (resolved during brainstorming)

- **Data source:** the user provides the raw NGSIM file locally (the canonical
  hosts — `data.transportation.gov`, PeMS — are unreachable from the dev
  container; only GitHub is). No runtime network dependency.
- **Segment:** NGSIM **I-80 (Emeryville)** — canonical stop-and-go wave data.
- **Target field:** reconstruct **density ρ(x,t)** via Edie's generalized
  definitions, on a **spatiotemporal window containing one dominant stop-and-go
  wave** so that ρ is monotone in x across it (monotonicity honest by disclosed
  window selection, not assumed for arbitrary corridors).
- **Raw handling:** raw CSV is **gitignored** (kept local). The small **derived
  artifact `.npz` is committed via Git LFS** (consistent with the paper figures'
  `*.pdf` LFS routing). CI and reruns depend only on the committed `.npz`.
- **Evaluation:** the dense Edie field (aggregating many vehicles ⇒ low noise) is
  the best-estimate **reference**; sparse **virtual loop detectors** (a few fixed
  x-positions, all times) are the model input. Metrics: dense **L¹/L²** vs the
  reference, **held-out-detector RMSE** (the cleanest metric — predict where you
  did not observe), and **admissibility violation**. Same four methods
  (hard_monotone / vanilla / soft / weight_clip), same equal-budget Optuna
  tune-once protocol as §6.2.

## Architecture

Four units, each independently testable, plugging into existing interfaces.

### 1. Preprocessing — `applications/pinn/data/ngsim.py`

Pure, offline. Reads the raw NGSIM CSV (10 Hz vehicle trajectories; columns
`Vehicle_ID, Frame_ID, Global_Time, Local_Y (position along road), v_Vel,
Lane_ID`), produces the committed `.npz`. NGSIM has **no density column** —
density is derived, which is the point of this step.

**Edie aggregation (trajectories → dense fields).** Lay a space-time grid of
cells over the section (default Δx ≈ 20 m, Δt ≈ 5 s). For each cell `A` of area
`|A| = Δx·Δt`, Edie's generalized definitions give
`d(A)` = total distance travelled by all vehicles while inside `A`,
`t(A)` = total time spent inside `A`, and then
**ρ(A) = t(A)/|A|**, **q(A) = d(A)/|A|**, **v = q/ρ**. Because NGSIM is sampled at
fixed 0.1 s, each consecutive trajectory sample-pair is a small segment
contributing `Δposition` to `d` and `≈0.1 s` to `t`, binned into the cell of its
midpoint:

```
for each vehicle, for each consecutive sample pair (y_k,t_k)->(y_{k+1},t_{k+1}):
    cell = grid_cell_of(midpoint_x, midpoint_t)
    dist[cell] += y_{k+1}-y_k;  time[cell] += t_{k+1}-t_k
rho = time / cell_area;  q = dist / cell_area          # nt × nx fields
```

Aggregating hundreds of vehicles per cell makes `rho` a low-noise best-estimate
reference; its residual noise is *real* driver behaviour (the whole motivation).

**Pipeline:** (a) parse + select a single lane (default) or lane-average;
(b) Edie-aggregate the **full** section → ρ, q fields; (c) run the **window scan**
(§1a) to pick the monotone window and crop to it; (d) calibrate the **fundamental
diagram** f(ρ) = q(ρ) on the window's (ρ, q) scatter — a triangular FD (free-flow
speed `v_f`, backward-wave speed `w`, jam density `ρ_jam`), Greenshields fallback
(`q = v_f·ρ(1−ρ/ρ_jam)`) if ill-conditioned; `flux_prime = dq/dρ`; (e) write
`data/ngsim-i80-wave.npz`: `x`, `t`, `rho` (nt×nx), `q`, `fd_params`, `sign_x`,
`monotonicity_defect`, plus provenance (segment, lane, window, source note).

CLI: `python -m applications.pinn.data.ngsim --raw <path> --out <npz>`.

### 1a. Window selection & monotonicity gate

Real I-80 has **several simultaneous stop-and-go waves**, so the full field is
generically **non-monotone in x**. We do not manufacture monotonicity; we *find*
the sub-region where the real field already is (nearly) monotone and *quantify*
it. This matters for **fairness**, not just honesty: residual non-monotonicity in
the reference is an irreducible error floor that handicaps the hard-monotone
model, so the window must be genuinely clean and the residual reported.

**Objective scan** (reuses `admissibility.violation`):

```
for each candidate window (x-range, t-range) over the Edie field:
    v  = Σ_t violation(rho[t, xslice], axis=x, sign)      # wrong-sign variation
    tv = Σ_t total |Δρ| along xslice                       # total variation
    monotonicity_defect = v / tv                            # 0 = perfectly monotone
pick the LARGEST window with monotonicity_defect < τ  (default τ = 0.05)
```

`sign_x` is inferred from the window's net density gradient. The chosen window's
`monotonicity_defect` is stored in the `.npz` and **reported verbatim in §6.5**
("the reference is N% monotone on the window"). The physically cleanest window is
usually the **onset of a single congestion wave** (free-flow downstream,
congested upstream, one front between) — the real-data analog of the synthetic
Riemann shock.

**Exploratory gate (a plan checkpoint, before the full experiment).** The first
step after aggregation is to run the scan, save the (x,t) speed/density heatmap,
and confirm a usable plain-x window exists (defect < τ, window large enough for a
meaningful sensor split). Only then do we build the experiment. If plain-x is too
dirty, pivot to the documented fallback: **wave-following coordinates** ξ = x −
c·t (estimate the dominant wave speed c from the congestion-band slope via
cross-correlation/Radon; enforce monotonicity in ξ, which a single wave leaves
stationary → a much wider clean window). If neither yields a usable window, we
downgrade the real-data claim to held-out-detector accuracy only and say so —
discovered early, not after building everything.

CLI: `python -m applications.pinn.data.ngsim --raw <path> --out <npz>`.

### 2. Problem plug-in — `applications/pinn/core/problems/traffic_real.py`

Registers `ngsim_wave` implementing the `Problem` protocol (`core/problems/base.py`),
constructed from the committed `.npz` (no raw dependency at run time):

- `domain` — the window bounds.
- `flux` / `flux_prime` — from the calibrated FD parameters.
- `admissibility()` — `AdmissibilitySpec(mask=(sign_x, 0))`.
- `ground_truth(x, t)` — bilinear interpolation of the loaded Edie field (this is
  what makes the dense L¹/L² eval work); returns the reference, not a synthetic
  exact solution.

Registered in `core/problems/__init__.py` beside `conservation`.

### 3. Virtual-detector sampling — extend `applications/pinn/core/sampling.py`

Add a **detector observation mode**: given the reference field, pick `n_detectors`
fixed x-positions (evenly spaced or provided), take **all** time samples at each →
observation coords/values. A disjoint `held_out_detectors` set is returned for
scoring. Noise comes from the data itself (`noise_std=0`); the existing random
scatter mode is untouched. This is the only change to shared sampling.

### 4. Experiment runner — reuse `experiments/headline.py` + `run.py`

No new runner. Add an `--observations {scatter,detectors}` flag threaded through
`RunConfig` → `run.py` (selecting the unit-3 sampling mode) and run the existing
tune-once + multi-seed `headline.py` with `--problem ngsim_wave --tier inverse
--observations detectors`:

- tunes each method once (equal Optuna budget) at the reference detector density;
- evaluates over seeds → IQM + bootstrap band of {L¹, L², held-out RMSE,
  admissibility violation};
- `run.py` adds **held-out-detector RMSE** to its metrics dict when the disjoint
  held-out set (unit 3) is present;
- writes `results/real-ngsim.json` (via `--out`).

## Data flow

```
raw NGSIM CSV (local, gitignored)
   └─ ngsim.py  ── Edie aggregation ─▶ window scan (defect < τ) ─▶ FD calibration ──▶  ngsim-i80-wave.npz (LFS)
                                                              │
core/problems/traffic_real.py  ◀── loads ──────────────────┘
   └─ Problem(ngsim_wave): domain, flux, admissibility, ground_truth=interp(field)
                                                              │
sampling.py (detector mode) ── virtual detectors + held-out ─┤
                                                              ▼
run.py (inverse) ── train {hard,vanilla,soft,weight_clip} ── eval ──▶ results/real-ngsim.json
                                                              │
figures.py ── reconstruction slice + metric bars (PDF+PNG) ──┘
```

## Figures

Via the existing `experiments/figures.py` (both PDF for LaTeX + PNG for preview):
- **Reconstruction slice** at a representative time: real field vs each method,
  virtual detectors overlaid (the money-shot analog on real data).
- **Metric bars / crossover** across detector density (or a small noise-injection
  sweep on top of the real field, if we want a stress axis).

## Testing

- `data/ngsim.py`: unit tests on Edie aggregation + FD calibration against a
  tiny hand-constructed trajectory fixture with a known answer; window-scan test
  on a synthetic field with a known monotone sub-region (correct window + defect).
- `traffic_real.py`: loads a **tiny synthetic `.npz` fixture** (committed, ~KB)
  and satisfies the `Problem` protocol; interpolation returns field values at grid
  nodes. CI needs **no raw NGSIM data**.
- `sampling.py` detector mode: correct count, disjoint held-out set, all-times
  coverage.
- Smoke test: full inverse path on the fixture problem for a few steps.

## Reproducibility

- RUNBOOK: the one manual step (obtain raw NGSIM I-80 → `data/raw/`), then the
  `ngsim.py` preprocess command, then the experiment + figure commands.
- Raw gitignored; derived `.npz` in LFS. `.gitattributes` gains `*.npz` (scoped to
  the data dir) alongside the existing `*.pdf` rule.

## Honest caveats (to appear verbatim in §6.5)

1. The reference is a dense **aggregation**, not true ground truth; **held-out
   detectors** are the cleaner metric and are reported first.
2. Monotonicity holds by **window selection** (one wave), chosen by an objective
   scan and reported as a `monotonicity_defect` number (the reference's own
   residual wrong-sign variation on the window) — disclosed; arbitrary corridors
   are not monotone in space. This is a scope statement, not a limitation of the
   method.
3. Real traffic is not exactly LWR, so the PDE residual carries model error — but
   **equally for all four methods**, so the comparison stays fair.

## Non-goals

- No PeMS / operational sparse estimation (follow-up).
- No higher-D, no multi-lane 2-D fields.
- No new training or model code — reuse existing trainers, builders, metrics.
- No raw NGSIM redistribution in the repo.
