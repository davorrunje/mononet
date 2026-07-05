# Deep monotonic residual — real-dataset accuracy

**Status: results pending the GPU search run.** This page reports whether the
now-trainable *deep* monotonic residual stacks (`MonoResidual` with
`sub_depth=2` skips — see [the residual construction](../concepts/monotonic-residual.md))
improve held-out test accuracy over the shallow tuned flavors, across the five
benchmark datasets.

## Question

Stage 1 showed residual skips make depth-32 monotone stacks *trainable* on
synthetic data. This study measures whether that trainability translates into
better test metrics on real tabular data, under the
[standard benchmark protocol](protocol.md) (5-fold CV model selection for the
small/medium datasets, single holdout for the large ones; mean ± std over all
final seeds; the test set is touched once).

A **null or negative result** — depth not improving, or mildly hurting, accuracy
on these small/medium tabular datasets — is an expected and reported outcome.
Stage 1 establishes the capability; Stage 2 measures whether it pays off.

## Flavors

Six flavors per dataset: `{switch, absolute} × {plain, residual, deep}`. The
**deep** flavor is a residual stack (`sub_depth=2`) whose depth is searched over
`{6, 10, 16}` blocks (effective ≈ 14 / 22 / 34 layers); plain/residual search
`depth ∈ [1, 4]`. All other hyperparameters share one search space, so depth is
the only structural difference between `residual` and `deep`.

## Results

_Results pending the GPU search run (see the reproduce command below). Test
metric is MSE for `auto`, RMSE for `blog`, accuracy for `heart`/`compas`/`loan`
(lower is better for MSE/RMSE; higher for accuracy)._

| dataset | metric | best shallow (mode) | deep (mode) | Δ | deep depth |
|---|---|---|---|---|---|
| auto | MSE | — | — | — | — |
| heart | accuracy | — | — | — | — |
| compas | accuracy | — | — | — | — |
| loan | accuracy | — | — | — | — |
| blog | RMSE | — | — | — | — |

## Reproduce

```
uv run --extra torch --group bench python -m benchmarks.search \
    --datasets auto,heart,compas,loan,blog
```

This runs all six flavors per dataset and writes
`benchmarks/results/phase2/<dataset>-<flavor>.json`. See
[`benchmarks/RUNBOOK-stage2.md`](https://github.com/davorrunje/mononet/blob/main/benchmarks/RUNBOOK-stage2.md)
for the full GPU run procedure.
