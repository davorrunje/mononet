# RUNBOOK — large-dataset monotonic-depth screen (GPU session)

Plumbing (the screen runner, gate, and report machinery) landed via CPU-session
PRs and is smoke-tested there on tiny synthetic bundles. This RUNBOOK is what
the GPU session runs to produce the real numbers and populate
[`docs/benchmarks/large-dataset-screen.md`](../docs/benchmarks/large-dataset-screen.md).

See the design spec:
`docs/superpowers/specs/2026-07-11-large-dataset-screen-design.md`.

## Data provenance

The committed Adult dataset (`benchmarks/data/adult/train_adult.csv.gz` and
`test_adult.csv.gz`) was prepared as follows:

1. Fetch the raw Adult dataset from UCI:

```python
import pandas as pd
from sklearn.datasets import fetch_openml

raw = fetch_openml(name="adult", version=1, as_frame=True, parser="auto").frame
raw.to_csv("adult_raw.csv", index=False)
```

2. Prepare via the mononet convention using `prepare_adult()`:

```python
import pandas as pd
import gzip

from benchmarks.datasets.prepare.adult import prepare_adult

raw = pd.read_csv("adult_raw.csv")
train, test = prepare_adult(raw)

train.to_csv(gzip.open("benchmarks/data/adult/train_adult.csv.gz", "wt"), index=False)
test.to_csv(gzip.open("benchmarks/data/adult/test_adult.csv.gz", "wt"), index=False)
```

The split is fixed and stratified (seed=0) on the `ground_truth` target. Both
frames have monotone columns (`education_num`, `hours_per_week`, `capital_gain`)
intact and numeric; categoricals are one-hot encoded.

## 1. Pull the LFS data

```bash
git lfs pull
```

This materializes all committed dataset files, including Adult and any others
that passed the screen in prior runs.

## 2. Run the max-size screen per dataset

For each dataset in the roster (starting with `adult`):

```bash
uv run --extra torch-gpu --group bench python -m benchmarks.large_screen_run \
  --dataset <name> \
  --out benchmarks/results/screen/<name>.json
```

This loads the dataset, runs the standard search for both `deep` and `shallow`
`mixed`-residual arms at the full train size, refits with multiple seeds on
the untouched test set, and writes a record with:
- `n_full`: the full train split size
- `deep_iqm` / `shallow_iqm`: the test-set IQM accuracies per arm
- `delta`: `deep_iqm - shallow_iqm`
- `delta_lo` / `delta_hi`: 95% bootstrap confidence interval on Δ
- `verdict`: `"ladder"` or `"standard"` (gate outcome)

**Expected cost:** variable per dataset size; budget accordingly per the design
spec's `_DATASET_PROTOCOL` table.

## 3. Collect records and generate the plot + table

Once all datasets have been screened, collect their records and generate the
summary plot and table:

```bash
uv run --group bench python -c "
import json
from pathlib import Path
from benchmarks._common.screen_report import render_screen_plot, screen_table

# Load all records
records = []
for dataset_name in ['adult']:  # add more as they are screened
    path = Path(f'benchmarks/results/screen/{dataset_name}.json')
    if path.exists():
        records.append(json.loads(path.read_text()))

# Render plot (PNG + PDF)
render_screen_plot(records, Path('docs/_static/large-dataset-screen.png'))

# Generate table
print(screen_table(records))
"
```

This writes both `docs/_static/large-dataset-screen.png` (embedded in the docs)
and a sibling `.pdf` (vector for the paper). Copy the printed table to the next
step.

## 4. Fill in the docs page

Edit [`docs/benchmarks/large-dataset-screen.md`](../docs/benchmarks/large-dataset-screen.md):

Replace the `{note}` placeholder block with:

- the plot: `![Δ per dataset](../_static/large-dataset-screen.png)`
- the summary table from step 3
- a brief interpretation: which datasets advanced to ladders (verdict =
  `"ladder"`)? Which joined the standard benchmark (verdict = `"standard"`)?
  Do the results align with the design hypothesis about dataset size and
  monotone-depth payoff?

## 5. Re-check the docs build

```bash
LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html
```

Expected: `build succeeded`, no warnings (the image reference in step 4 must
resolve now that `docs/_static/large-dataset-screen.png` exists).

## 6. Commit

Commit the results JSON, the generated PNG, the filled-in docs page, and the
plot PDF together so the docs page and its data stay in sync:

```bash
git add benchmarks/results/screen/*.json docs/_static/large-dataset-screen.png \
  docs/_static/large-dataset-screen.pdf docs/benchmarks/large-dataset-screen.md
git commit -m "bench(results): large-dataset screen — depth verdict per dataset"
```
