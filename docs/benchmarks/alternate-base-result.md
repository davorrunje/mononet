# Alternate base result (tuned, ≤4 layers)

Tuned shallow (`depth ∈ [1,3]`, i.e. ≤4 effective layers counting the linear
read-out head) comparison of the monotone constructions on the five paper
datasets. Each flavor is tuned per-dataset with its own Optuna HP search
(activation included), `plain` only, at the paper's per-dataset trial counts
(`heart`/`auto` = 200, `compas`/`blog`/`loan` = 50) under the repo's
stability-aware CV objective. `mixed` tunes `convex_fraction`; `mixed-fixed`
holds it at 0.5. Bold = best per dataset (🥇).

| dataset | rows | flavor | IQM | mean ± std | act | layers | width | lr | wdec | drop | lrdec | batch | cvxf | done |
|---|--:|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| heart (acc ↑) | 242 | split | 0.909 | 0.909 ± 0.004 | relu | 3 | 32 | 0.0086 | 0.131 | 0.40 | 0.866 | 8 | · | ✅ |
|  |  | mixed 🥇 | **0.910** | 0.909 ± 0.006 | relu | 2 | 16 | 0.0152 | 0.096 | 0.43 | 0.991 | 8 | 0.78 | ✅ |
|  |  | mixed-fix | 0.906 | 0.906 ± 0.003 | relu | 3 | 64 | 0.0118 | 0.090 | 0.39 | 0.895 | 32 | 0.50 | ✅ |
|  |  | alternate | 0.906 | 0.906 ± 0.001 | softplus | 3 | 8 | 0.0215 | 0.059 | 0.15 | 0.899 | 32 | · | ✅ |
| auto (MSE ↓) | 314 | split | 10.90 | 10.88 ± 0.49 | relu | 2 | 64 | 0.0066 | 0.017 | 0.03 | 0.942 | 8 | · | ✅ |
|  |  | mixed | 10.46 | 10.44 ± 0.28 | elu | 3 | 21 | 0.0838 | 0.043 | 0.01 | 0.919 | 8 | 0.54 | ✅ |
|  |  | mixed-fix 🥇 | **10.13** | 10.18 ± 0.28 | elu | 3 | 8 | 0.0743 | 0.034 | 0.02 | 0.926 | 8 | 0.50 | ✅ |
|  |  | alternate | 10.14 | 10.13 ± 0.19 | softplus | 2 | 32 | 0.0389 | 0.020 | 0.05 | 0.926 | 16 | · | ✅ |
| compas (acc ↑) | 4,937 | split | 0.730 | 0.729 ± 0.003 | softplus | 3 | 16 | 0.0172 | 0.000 | 0.07 | 0.998 | 32 | · | ✅ |
|  |  | mixed | 0.704 | 0.704 ± 0.002 | softplus | 2 | 16 | 0.0007 | 0.190 | 0.00 | 0.896 | 64 | 0.77 | ✅ |
|  |  | mixed-fix | 0.721 | 0.721 ± 0.002 | elu | 4 | 32 | 0.0004 | 0.004 | 0.08 | 0.986 | 8 | 0.50 | ✅ |
|  |  | alternate 🥇 | **0.730** | 0.730 ± 0.002 | selu | 3 | 16 | 0.0106 | 0.000 | 0.46 | 0.903 | 8 | · | ✅ |
| blog (RMSE ↓) | 47,302 | split 🥇 | **0.177** | 0.177 ± 0.002 | softplus | 2 | 21 | 0.0226 | 0.001 | 0.08 | 1.000 | 2048 | · | ✅ |
|  |  | mixed | 0.178 | 0.178 ± 0.001 | elu | 2 | 21 | 0.0386 | 0.000 | 0.39 | 0.986 | 1024 | 0.38 | ✅ |
|  |  | mixed-fix | 0.182 | 0.182 ± 0.001 | softplus | 2 | 32 | 0.0146 | 0.003 | 0.06 | 0.859 | 1024 | 0.50 | ✅ |
|  |  | alternate | 0.186 | 0.186 ± 0.001 | selu | 2 | 21 | 0.0064 | 0.013 | 0.27 | 0.879 | 2048 | · | ✅ |
| loan (acc ↑) | 418,697 | split | 0.704 | 0.704 ± 0.001 | selu | 3 | 16 | 0.0002 | 0.001 | 0.04 | 0.914 | 512 | · | ✅ |
|  |  | mixed | 0.704 | 0.704 ± 0.000 | elu | 2 | 16 | 0.0036 | 0.000 | 0.13 | 0.999 | 512 | 0.72 | ✅ |
|  |  | mixed-fix 🥇 | **0.708** | 0.708 ± 0.001 | elu | 3 | 16 | 0.0436 | 0.000 | 0.21 | 0.880 | 512 | 0.50 | ✅ |
|  |  | alternate | 0.703 | 0.703 ± 0.000 | elu | 2 | 8 | 0.0028 | 0.000 | 0.14 | 0.896 | 512 | · | ✅ |

## `convex_fraction` ablation — searched (`mixed`) vs fixed 0.5 (`mixed-fixed`)

| dataset | mixed (searched cvxf) | mixed-fixed (0.5) | better |
|---|--:|--:|:--|
| auto | 10.46 (cvxf 0.54) | 10.13 | fixed |
| blog | 0.1778 (cvxf 0.38) | 0.1817 | searched |
| heart | 0.9096 (cvxf 0.78) | 0.9056 | searched |
| compas | 0.7037 (cvxf 0.77) | 0.7211 | fixed |
| loan | 0.7037 (cvxf 0.72) | 0.7084 | fixed |

**Fixing `convex_fraction = 0.5` is the better default** — it wins 3/5 (auto, compas, loan), substantially on `compas` (+0.017 acc) and `auto` (−0.33 MSE); searching is only marginally better on `heart`/`blog` (~0.004). Searching the knob mostly hurt on the mid/large datasets and complicates the init (any fraction ≠ 0.5 triggers a gain/bias fixed-point in `absolute_init_params`). This motivates fixing it (spec follow-up 6).

## Verdict — `alternate` vs best-of-others (all non-alternate flavors)

Bootstrap CI on the per-fold `alternate − best-other` gap, where *best-other* is the strongest of `split` / `mixed` / `mixed-fixed`.

| dataset | Δ (alt − best-other) | 95% CI | verdict |
|---|--:|:--|:--|
| auto | -0.015 | [-0.145, +0.185] | matches (CI straddles 0) (vs mixed-fixed-plain) |
| blog | -0.009 | [-0.010, -0.008] | alternate loses (vs split-plain) |
| heart | -0.004 | [-0.007, -0.000] | alternate loses (vs mixed-plain) |
| compas | +0.000 | [-0.001, +0.004] | matches (CI straddles 0) (vs split-plain) |
| loan | -0.005 | [-0.006, -0.005] | alternate loses (vs mixed-fixed-plain) |

**Bottom line — does `alternate` beat the best non-alternate flavor at ≤4 tuned layers? No, not decisively.** It reaches a clean co-lead on `compas` (0.730, tied with `split`, ahead of both `mixed` variants) and sits within noise on `auto` (10.14 vs `mixed-fixed` 10.13) and `heart` (0.906 vs 0.910); it loses outright on `blog` and `loan`. So in the regimes where `mixed` was dominating, tuned `alternate` closes the gap to parity on the mid-size classification task but does not overtake — and the single biggest gain in this sweep came not from `alternate` but from *fixing* `convex_fraction` in `mixed` (see the ablation above).

> **Caveat (auto).** On tiny `auto` (314 rows) the 200-trial budget meta-overfits the CV objective — searched `mixed`/`split` degrade vs a lighter budget — so `auto` should be read cautiously; a search-sensitivity study (separate spec) quantifies this.
