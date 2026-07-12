# RUNBOOK — monotone-depth synthetic probe (GPU session)

Plumbing (the probe run, report machinery) landed via CPU-session PRs and is
smoke-tested there on tiny synthetic bundles. This RUNBOOK is what the GPU
session runs to produce the real numbers and populate
[`docs/benchmarks/monotone-depth-probe.md`](../docs/benchmarks/monotone-depth-probe.md).

See the design spec:
`docs/superpowers/specs/2026-07-12-monotone-depth-synthetic-probe-design.md`.

## 1. Run the probe per (kind, c)

For each combination of family and complexity knob:

```bash
MONONET_TORCH_DEVICE=cuda:0 uv run --extra torch-gpu --group bench \
  python -m benchmarks.monotone_depth_probe_run \
  --kinds additive,teacher_relu,teacher_elu,lattice \
  --cs 1,2,4,8 \
  --out benchmarks/results/depth-probe/probe.json
```

This sweeps the three families (additive, teacher, lattice) across all
complexity knobs in a single run, executing one `(kind, c)` process per GPU
slot with `n_jobs=1` (never threaded Optuna — see the [large-dataset
screen](large-dataset-screen.md) deadlock lesson). For each `(kind, c)`:

- Generates a synthetic monotone-regression bundle via `synth_monotone(kind, c)`
  (6-dimensional, dense sampling, noise-free)
- Runs the standard search for both `deep` (depth ∈ {6, 10, 16}) and `shallow`
  (depth ∈ {1, 4}) `absolute`-residual arms
- Refits with 8 test seeds and computes per-arm MSE IQM (+ raw per-seed values
  for the report's bootstrap)
- Writes a record with `kind`, `c`, `deep_mse_iqm`, `shallow_mse_iqm`,
  `deep_values`, and `shallow_values`

All records are collected into `benchmarks/results/depth-probe/probe.json`.

**Expected cost:** ~4 hours for all 12 (kind, c) combos at standard tuning
(15 trials, 2 search seeds, 8 final seeds, 30 epochs). Budget accordingly.

## 2. Collect records and generate the plot + table

Once the probe completes, collect all records and render the summary plot and
table:

```bash
uv run --group bench python -c "
import json
from pathlib import Path
from benchmarks._common.depth_probe_report import delta_by_c, probe_table, render_probe_plot

# Load probe records
path = Path('benchmarks/results/depth-probe/probe.json')
records = json.loads(path.read_text())

# Compute Δ(c) with bootstrap bands
delta_rows = delta_by_c(records)

# Render plot (PNG + PDF)
render_probe_plot(delta_rows, Path('docs/_static/monotone-depth-probe.png'))

# Generate table
print(probe_table(delta_rows))
"
```

This writes both `docs/_static/monotone-depth-probe.png` (embedded in the docs)
and a sibling `.pdf` (vector for the paper), and prints a markdown table with
one row per (kind, c) showing deep MSE, shallow MSE, and Δ with its 95%
bootstrap confidence interval. Copy the printed table to the next step.

## 3. Fill in the docs page

Edit [`docs/benchmarks/monotone-depth-probe.md`](../docs/benchmarks/monotone-depth-probe.md):

Replace the `{note}` placeholder block with:

- the plot: `![Δ(c) per family](../_static/monotone-depth-probe.png)`
- the summary table from step 2
- a brief interpretation: Do the Δ(c) curves support **H-strong** (flat at zero
  across families) or **H-weak** (rising with `c`)? Which families show signals
  (Δ > 0, confident)?

## 4. (Optional) Iso-parameter frontier

If any family/c shows a signal in the fixed deep/shallow bands, run the
iso-parameter frontier sweep for those cases only. This locates the error
minimum along an equal-parameter-count curve (varying depth and width). Details
in the [design spec](https://github.com/davorrunje/mononet/blob/main/docs/superpowers/specs/2026-07-12-monotone-depth-synthetic-probe-design.md)
section 3.3.

## 5. Re-check the docs build

```bash
LC_ALL=C.UTF-8 LANG=C.UTF-8 uv run sphinx-build -W docs docs/_build/html
```

Expected: `build succeeded`, no warnings (the image reference in step 3 must
resolve now that `docs/_static/monotone-depth-probe.png` exists).

## 6. Commit

Commit the results JSON, the generated PNG, the filled-in docs page, and the
plot PDF together so the docs page and its data stay in sync:

```bash
git add benchmarks/results/depth-probe/probe.json docs/_static/monotone-depth-probe.png \
  docs/_static/monotone-depth-probe.pdf docs/benchmarks/monotone-depth-probe.md
git commit -m "bench(results): monotone-depth probe — Δ(c) per family"
```
