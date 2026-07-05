# Stage 2 GPU run — deep monotonic residual accuracy

This runbook executes the Stage-2 accuracy study on a GPU machine
(`gpu-torch` devcontainer, 5090 / Blackwell sm_120). The plumbing (deep flavor,
CLI, docs skeleton) is already merged; this run produces the numbers.

## 0. Environment

Open the repo in the **`gpu-torch`** devcontainer flavor (see
`.devcontainer/`). Then:

```bash
uv sync --extra torch --group bench
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True` and the 5090 device name.

## 1. Run the full search

Six flavors per dataset (`{switch,absolute} × {plain,residual,deep}`), default
per-dataset budgets (`auto/heart/compas`: 50 trials, 10 seeds, 5-fold CV;
`loan/blog`: 25 trials, 5 seeds, single holdout):

```bash
uv run --extra torch --group bench python -m benchmarks.search \
    --datasets auto,heart,compas,loan,blog \
    --storage-dir benchmarks/results/deep-residual-accuracy/studies
```

Notes:
- Only `auto` had results under the standard protocol; heart/compas/loan/blog
  shallow flavors are (re)generated here too.
- The `sub_depth=2` default (merged in #67) makes the *prior* `auto` residual
  numbers stale — this run regenerates all `auto` flavors, so the whole table
  is internally consistent.
- `--storage-dir` writes resumable Optuna `.db` files; these are git-ignored
  (never commit `*.db`). Use `--n-jobs` to parallelize trials on the GPU box.
- To smoke-test the plumbing first: append `--smoke` (tiny 2-trial/2-fold run).

Outputs: `benchmarks/results/phase2/<dataset>-<flavor>.json` for all 5 × 6 = 30
files (four `auto-*` are overwritten; the rest are new).

## 2. Commit the results

```bash
git checkout -b feat/deep-residual-accuracy-results
git add benchmarks/results/phase2/*.json
git commit -S -m "bench(results): Stage 2 deep-vs-shallow accuracy (all 5 datasets, 6 flavors)"
```

(Confirm no `*.db`/`*.jsonl` are staged.)

## 3. Fill the docs table

Edit `docs/benchmarks/deep-residual-accuracy.md` — replace the placeholder
`—` cells in the Results table. For each dataset:
- **metric**: MSE (`auto`), RMSE (`blog`), accuracy (`heart`/`compas`/`loan`).
- **best shallow (mode)**: the better of the four shallow flavors
  (`{switch,absolute}-{plain,residual}`) by `test_mean` (min for MSE/RMSE, max
  for accuracy); note which mode won.
- **deep (mode)**: the better of `{switch,absolute}-deep` by `test_mean`.
- Report each as `test_mean ± test_std`; **Δ** = deep − best-shallow (sign per
  the metric's direction — note whether deep helped).
- **deep depth**: `best_params["depth"]` of the reported deep flavor.

Remove the "Status: results pending" banner and the "Results pending" italic
note once filled. Then:

```bash
./tools/build-docs.sh   # expect: build succeeded, no new warnings
```

## 4. Open the follow-up PR

```bash
git push -u origin feat/deep-residual-accuracy-results
gh pr create --base main \
    --title "bench: Stage 2 deep-vs-shallow monotonic residual accuracy" \
    --body "Fills the deep-residual-accuracy results table from the GPU search run. Closes the Stage 2 follow-up."
```

All commits must be signed. If tool-driven signing is flaky, commit unsigned
and re-sign before push:
`git rebase --exec "git commit --amend --no-edit -n -S" $(git merge-base main HEAD)`.
