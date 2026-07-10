# RUNBOOK — loan size-ladder (GPU session)

Plumbing (the `subsample_train` helper, `loan_size_ladder_run.py`, and
`benchmarks/_common/size_ladder_report.py`) landed via CPU-session PRs and is
smoke-tested there on a tiny synthetic bundle. This RUNBOOK is what the GPU
session runs to produce the real numbers and populate
[`docs/benchmarks/loan-size-ladder.md`](../docs/benchmarks/loan-size-ladder.md).

See the design spec:
`docs/superpowers/specs/2026-07-10-loan-size-ladder-design.md`.

## 1. Run the ladder

```bash
uv run --extra torch-gpu --group bench python -m benchmarks.loan_size_ladder_run
```

This loads the `loan` dataset, runs the full N-ladder
(`5_000, 15_000, 45_000, 135_000, <full>`) for both the `shallow` and `deep`
`absolute`-residual arms, and writes the committed results JSON to
`benchmarks/results/size-ladder/loan.json`.

**Expected cost:** the ladder is cheap at small N (each rung tunes and refits
on a subsample) and gets expensive as N grows — only the **top 1–2 rungs**
(135,000 and the full ~419k-row set) are heavy; the full-N rung mirrors the
`loan` run from PR #72's deep-residual-accuracy benchmark, so budget
accordingly.

## 2. Generate the plot

```bash
uv run --group bench python -c "
import json
from pathlib import Path
from benchmarks._common.size_ladder_report import render_plot

records = json.load(open('benchmarks/results/size-ladder/loan.json'))
render_plot(records, Path('docs/_static/loan-size-ladder.png'))
"
```

## 3. Generate the per-N table

```bash
uv run --group bench python -c "
import json
from benchmarks._common.size_ladder_report import delta_by_n

records = json.load(open('benchmarks/results/size-ladder/loan.json'))
for row in delta_by_n(records):
    print(row)
"
```

Format the printed rows into a Markdown table with columns `N`, `shallow IQM`,
`deep IQM`, `Δ`, and the `[delta_lo, delta_hi]` bootstrap band.

## 4. Fill in the docs page

Edit [`docs/benchmarks/loan-size-ladder.md`](../docs/benchmarks/loan-size-ladder.md):

- Replace the `{note}` placeholder block with:
  - the plot, e.g. `![Δ IQM vs N](../_static/loan-size-ladder.png)`
  - the per-N table from step 3
  - a short interpretation: does `Δ(N)` cross zero and stay positive past some
    N, consistent with "deep wins once the dataset is large enough"? Or does it
    stay flat/negative, which would undercut the PR #72 hypothesis?
- Remove the `<!-- The GPU run replaces this block ... -->` comment once the
  block is replaced.

## 5. Re-check the docs build

```bash
LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html
```

Expected: `build succeeded`, no warnings (the image reference in step 4 must
resolve now that `docs/_static/loan-size-ladder.png` exists).

## 6. Commit

Commit the results JSON, the generated PNG, and the filled-in docs page
together so the docs page and its data stay in sync.
