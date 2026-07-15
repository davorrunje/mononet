# HP-search sensitivity curves — design

## Context

The tuned base-result bake-off (PR #111,
[`2026-07-14-alternate-base-result-design.md`](2026-07-14-alternate-base-result-design.md))
exposed that per-flavor verdicts are **budget-dependent**. Two concrete
observations motivated this spec:

- `alternate` went from winning 0/5 datasets at a 25–50-trial budget to
  co-leading 2/5 once studies were topped up to the paper's per-dataset trial
  counts (`heart`/`auto` = 200, `compas`/`blog`/`loan` = 50).
- On tiny `auto` (314 rows), the 200-trial budget **meta-overfits** the CV
  objective: the searched `mixed`/`split` incumbents' held-out test metric
  degrades relative to a lighter budget, even as the CV objective keeps
  improving.

A single reported number therefore hides whether a flavor's search had
*saturated* (the number is stable) or was still *climbing* (the bake-off
verdict would flip with more trials), and whether the incumbent is
*generalising* or *meta-overfitting*. This spec adds a committed diagnostic
that answers both, reconstructed from the Optuna storage the base-result run
already committed — **no re-training of the search**.

**Prerequisite.** This deliverable reads the committed Optuna storage DBs
under `benchmarks/results/alternate-base/studies/{dataset}-{flavor}.db`, which
land on `main` when PR #111 merges. Implement after #111 merges (or rebase the
implementation branch onto it). The spec and plan themselves carry no code
dependency and can be reviewed independently.

## Goal

For each `(dataset, flavor)` study, plot how the tuned result evolves with the
number of Optuna trials, and quantify whether the search budget was enough.
Ship it as a committed, reproducible benchmark diagnostic (module + CLI + docs
page + committed figures), a sibling to the existing size-ladder report
([`size_ladder_report.py`](../../../benchmarks/_common/size_ladder_report.py)).

## What the Optuna storage does and does not contain

Established by reading `benchmarks/_common/search.py`:

- Each trial's `value` **is** the stability-aware CV objective that TPE
  optimized (`mean ± std` over `n_splits` folds × `search_seeds` seeds; the
  variance-penalized bound). It is stored per trial in the DB, in trial order.
- The **held-out test metric is *not* stored per trial** — it is computed only
  in `final_eval`, once, for the best params. So a test-vs-trial curve cannot
  be read directly from the existing DBs.
- Each study is a **single TPE trajectory** (one sampler seed = the study
  `seed`). There is no ensemble of independent searches, so a sampler-seed
  confidence band is not available from committed data without a re-run.

These three facts fix the design below.

## Design

### Curve A — search-objective best-so-far (primary, zero re-train)

For each study, read completed trials in trial-number order and compute the
cumulative best objective `b(t)` (running `min` for minimize metrics, `max`
for maximize). `b(t)` is monotone by construction. This is the search's
"meta-training" convergence curve and is free from the DB.

**Saturation point.** Let `G = |b(T) − b(1)|` be the total gain over the
`T`-trial budget. Define

```
t*(p) = min { t : |b(T) − b(t)| ≤ (1 − p) · G }
```

i.e. the smallest trial count that reaches a fraction `p` of the eventual
gain. Report `t*(0.99)` per `(dataset, flavor)`. `t* ≪ T` ⇒ the budget was
more than enough (saturated); `t* ≈ T` ⇒ still climbing (budget too small,
verdict unreliable). `G ≈ 0` (no improvement after trial 1) is reported as
`t* = 1`. This metric needs Curve A only — no re-train.

### Curve B — test metric of the running incumbent (meta-generalization)

The diagnostic that reveals meta-overfitting. At each trial `t`, the incumbent
is the best-objective trial in `1..t`; plot its **test** metric against `t`.
Unlike Curve A this is *not* monotone — it can turn back up (regression) or
down (classification) as the search overfits the CV objective (the `auto`
case).

The incumbent changes only at the **best-so-far changepoints** — a handful of
trials per study, not all `T`. So Curve B is reconstructed by re-evaluating
**only the distinct incumbent configs** on the held-out test set via the
existing `final_eval`, and drawing a step function that holds each incumbent's
test metric until the next changepoint. This is a **bounded re-evaluation**
(number of changepoints, typically ≪ 20), explicitly *not* a re-run of the
search, and every re-evaluated config count is logged so the cost is never
silently hidden.

`final_eval` needs the reconstructed `BenchmarkConfig` for a stored trial's
params. The params are in `trial.params`; the flavor's fixed knobs (mode,
residual, `embed_layers`, `alt_init`, fixed `convex_fraction`) are recovered
from the study name and the base-result run's settings, exactly as
`search.final_eval` already does for the best trial.

### Prospective hook — per-trial test logging (optional, off by default)

So that *future* sensitivity runs get Curve B for free (no incumbent
re-eval), add an optional `log_test_trajectory: bool = False` flag to
`search()`. When set, after each trial the objective records the trial's
held-out test metric as `trial.set_user_attr("test_metric", …)`. It is **never
read back into selection** — purely a diagnostic attribute. It is off by
default because it adds a full extra evaluation per trial (roughly doubling
search cost). Existing committed studies lack the attribute, so the
retrospective incumbent-reconstruction above remains the method for this
deliverable; the reconstructor prefers the stored attribute when present and
falls back to re-eval when absent.

### Bands and TPE-vs-random reference (scoped out of the core)

A proper confidence band needs `K` independent-sampler-seed studies per
`(dataset, flavor)` — a bounded re-run we do not have and do not run here.
A *random-order reference band* (bootstrap over trial order to show the
best-so-far distribution random search would achieve, contrasting TPE's
guidance) is computable from committed data with no re-train, but it answers a
different question and risks being misread as a sampler-seed CI. Both are
recorded as follow-ups, not built in the core deliverable, to keep it focused.

### Committed artifacts

- **`benchmarks/_common/sensitivity_report.py`** — pure analysis + render:
  - `best_so_far(study, lower) -> list[float]` — Curve A trajectory.
  - `saturation_trial(traj, lower, p=0.99) -> int` — `t*(p)`.
  - `incumbent_changepoints(study, lower) -> list[tuple[int, dict]]` —
    `(trial_index, params)` at each best-so-far improvement.
  - `render_plot(studies, out_path)` — faceted figure, one facet per dataset,
    one line per flavor for Curve A (x = trial index, y = objective), mirroring
    `size_ladder_report.render_plot` (Agg backend, PNG + PDF, publication
    style, no title). Curve B overlaid as a second panel row.
- **`benchmarks/sensitivity.py`** — CLI: load a storage dir's DBs, render the
  figure, and print a Markdown **saturation table** (`dataset | flavor |
  trials | t*(0.99) | saturated? | # incumbents re-evaluated`).
- **`docs/benchmarks/hp-search-sensitivity.md`** — the figure + saturation
  table + interpretation, with `auto`'s meta-overfitting called out explicitly.
- Committed figure output (PNG for docs, PDF for a paper), regenerated by the
  CLI and checked in like the size-ladder figure.

## Testing

- `best_so_far` / `saturation_trial` on synthetic monotone and flat
  trajectories (known `t*`); direction (`lower`) both ways.
- `incumbent_changepoints` returns exactly the improving trials, in order, on
  a hand-built fake study.
- `render_plot` smoke test: writes non-empty `.png` and `.pdf` (Agg), mirroring
  the size-ladder report's render test.
- The bounded re-eval path is covered by a small fake study (2–3 incumbents)
  asserting `final_eval` is called once per distinct incumbent, not per trial.

## Follow-ups

1. Sampler-seed confidence band: `K` independent-seed studies per
   `(dataset, flavor)` → real CI on the trajectory (bounded re-run).
2. Random-order reference band (TPE vs random search) as an optional overlay.
3. Once `log_test_trajectory` is standard for dedicated sensitivity runs, drop
   the incumbent re-eval path in favor of the stored `test_metric` attribute.
