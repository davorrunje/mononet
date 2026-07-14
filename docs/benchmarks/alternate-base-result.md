# Alternate base result (tuned, ≤4 layers)

This page reports the tuned shallow (`depth ∈ [1,3]`, i.e. ≤4 effective layers
counting the linear read-out head) `alternate` flavor against the best of the
existing `split`/`mixed` flavors, on the five paper datasets. Each flavor is
tuned per-dataset with its own Optuna HP search (including the activation),
`plain` only (no residual arm), following the
[standard benchmark protocol](protocol.md). The verdict per dataset is a
bootstrap CI on `alternate − best-of-{split, mixed}`: "alternate helps" if the
CI lies strictly on the better side of zero, "matches" if it straddles zero,
"loses" otherwise.

| dataset | rows | flavor | IQM | mean ± std | act | layers | width | lr | wdec | drop | lrdec | batch | cvxf | done |
|---|--:|---|--:|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| heart (acc ↑) | 242 | split | 0.909 | 0.909 ± 0.004 | relu | 3 | 32 | 0.0086 | 0.131 | 0.40 | 0.866 | 8 | · | ✅ |
|  |  | mixed 🥇 | **0.910** | 0.909 ± 0.006 | relu | 2 | 16 | 0.0152 | 0.096 | 0.43 | 0.991 | 8 | 0.78 | ✅ |
|  |  | alternate | 0.906 | 0.906 ± 0.001 | softplus | 3 | 8 | 0.0215 | 0.059 | 0.15 | 0.899 | 32 | · | ✅ |
| auto (MSE ↓) | 314 | split | 10.90 | 10.88 ± 0.49 | relu | 2 | 64 | 0.0066 | 0.017 | 0.03 | 0.942 | 8 | · | ✅ |
|  |  | mixed | 10.46 | 10.44 ± 0.28 | elu | 3 | 21 | 0.0838 | 0.043 | 0.01 | 0.919 | 8 | 0.54 | ✅ |
|  |  | alternate 🥇 | **10.14** | 10.13 ± 0.19 | softplus | 2 | 32 | 0.0389 | 0.020 | 0.05 | 0.926 | 16 | · | ✅ |
| compas (acc ↑) | 4,937 | split | 0.730 | 0.729 ± 0.003 | softplus | 3 | 16 | 0.0172 | 0.000 | 0.07 | 0.998 | 32 | · | ✅ |
|  |  | mixed | 0.704 | 0.704 ± 0.002 | softplus | 2 | 16 | 0.0007 | 0.190 | 0.00 | 0.896 | 64 | 0.77 | ✅ |
|  |  | alternate 🥇 | **0.730** | 0.730 ± 0.002 | selu | 3 | 16 | 0.0106 | 0.000 | 0.46 | 0.903 | 8 | · | ✅ |
| blog (RMSE ↓) | 47,302 | split 🥇 | **0.177** | 0.177 ± 0.002 | softplus | 2 | 21 | 0.0226 | 0.001 | 0.08 | 1.000 | 2048 | · | ✅ |
|  |  | mixed | 0.178 | 0.178 ± 0.001 | elu | 2 | 21 | 0.0386 | 0.000 | 0.39 | 0.986 | 1024 | 0.38 | ✅ |
|  |  | alternate | 0.186 | 0.186 ± 0.001 | selu | 2 | 21 | 0.0064 | 0.013 | 0.27 | 0.879 | 2048 | · | ✅ |
| loan (acc ↑) | 418,697 | split 🥇 | **0.704** | 0.704 ± 0.001 | selu | 3 | 16 | 0.0002 | 0.001 | 0.04 | 0.914 | 512 | · | ✅ |
|  |  | mixed | 0.704 | 0.704 ± 0.000 | elu | 2 | 16 | 0.0036 | 0.000 | 0.13 | 0.999 | 512 | 0.72 | ✅ |
|  |  | alternate | 0.703 | 0.703 ± 0.000 | elu | 2 | 8 | 0.0028 | 0.000 | 0.14 | 0.896 | 512 | · | ✅ |

## Verdict — alternate vs best-of-others

| dataset | Δ (alt − best-other) | 95% CI | verdict |
|---|--:|:--|:--|
| heart | -0.004 | [-0.007, -0.000] | alternate loses (vs mixed-plain) |
| auto | +0.318 | [+0.156, +0.479] | alternate **beats** best-of-others (vs mixed-plain) |
| compas | +0.000 | [-0.001, +0.004] | matches (CI straddles 0) (vs split-plain) |
| blog | -0.009 | [-0.010, -0.008] | alternate loses (vs split-plain) |
| loan | -0.001 | [-0.001, -0.000] | alternate loses (vs split-plain) |

> **Caveats.** (1) On tiny **auto** (314 rows), pushing the search to 200 trials
> meta-overfits the CV objective: the incumbents degrade vs a lighter budget, so
> alternate's lead there is not a clean win. (2) The **mixed** rows tune
> `convex_fraction` (`cvxf`); a follow-up adds a fixed-`cvxf=0.5` mixed variant as a
> separate flavor to isolate whether searching it helps.
