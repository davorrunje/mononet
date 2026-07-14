# Alternate base result: does tuned `alternate` beat the incumbents at ≤4 plain layers?

**Status:** design
**Date:** 2026-07-14
**Depends on:** #109 (`mode="alternate"` + `prev=` composition init), merged.

## Goal

Establish the **base result** for the `alternate` monotone construction: with
per-flavor Optuna hyperparameter tuning at **shallow plain depth (≤4 layers)**,
does a tuned `alternate` stack **beat the best non-alternate flavor** on the
paper's five benchmark datasets — i.e. is `alternate` helpful in exactly the
regime where `mixed` was dominating?

This produces a drop-in replacement for the README "Benchmark results" table,
with an `alternate` row added per dataset and the bold (best) recomputed across
all flavors.

## Motivation

The fixed-HP depth ablation ([`flavor_ablation.py`]) already showed, on `heart`
and `auto`, that at a *shared* learning rate `mixed` wins shallow while
`mixed`/`split` diverge deep and `alternate` stays stable. But that comparison
holds HPs constant, so it cannot answer whether a *tuned* `alternate` matches or
beats a *tuned* `mixed` at the shallow depths that actually win. The depth-8 LR
mini-sweep hinted `alternate` pulls ahead once LR is tuned, but only at depth 8.
The open question — and the one the README table implicitly answers for
`split`/`mixed` — is the **tuned shallow head-to-head**. This spec answers it and
adds `alternate` to the canonical table.

Depth is a deprioritized follow-up: the evidence already suggests depth does not
help, so the base result is fixed at ≤4 plain layers.

## Scope

**In scope**
- Three flavors, **plain topology only**: `split-plain`, `mixed-plain`,
  `alternate-plain`.
- Depth `∈ [1, 3]` → ≤4 layers under the theorem's layer count (see *Layer
  counting*); a capped shallow band.
- Activation searched over `{relu, elu, softplus, selu}` (was pinned to `elu`).
- All five paper datasets: `heart`, `auto`, `compas`, `loan`, `blog`.
- Per-flavor Optuna HP search + multi-seed final eval, reusing the existing
  `search()` / `final_eval()` machinery and per-dataset budgets.
- README-format results table with a bootstrap-delta verdict.

**Out of scope (follow-ups)**
- Residual and deep bands (`residual + alternate` is unimplemented; deep is
  deprioritized). The base result is plain-only.
- The `flavor_ablation.py` fixed-HP depth grid — set aside, neither extended nor
  deleted.
- Re-running the paper-reproduction / Stage-2 tables (their `elu`-fixed semantics
  are preserved; see backward-compat below).

## Architecture — Approach A (extend the existing pipeline in place)

Reuse the proven Optuna pipeline end to end; add no parallel modules (the
current repo already carries a second launcher/report from the depth study, and
we do not want a third).

```
search_spaces.suggest_config  ──►  search.search() ──►  per-flavor result JSON
        (+ alternate, + activation)        │ (Optuna study, resumable via storage)
                                           ▼
                              search.final_eval()  (best HPs → multi-seed test eval)
                                           │
                                           ▼
                     stage2_launch.py  (multi-GPU fan-out, --storage-dir resume)
                                           │
                                           ▼
                     make_tables.render()  (README table + bootstrap verdict)
```

### Change 1 — `search_spaces.suggest_config`

- Widen `mode` to `Literal["split", "mixed", "alternate"]`.
- When `mode == "alternate"`: set the returned config's `alt_init="composition"`
  and **do not** sample `convex_fraction` (alternate derives its own phases).
- Add a keyword `search_activation: bool = False`.
  - `False` (default): keep `activation="elu"` — paper-reproduction, Stage-2, and
    the existing `test_search_spaces` assertions are byte-for-byte unchanged.
  - `True`: `activation = trial.suggest_categorical("activation", ["relu", "elu",
    "softplus", "selu"])`.
- Add a keyword `max_depth: int = 4` (default preserves the existing `[1, 4]`
  band); `depth = trial.suggest_int("depth", 1, max_depth)`. This run passes
  `max_depth=3` (≤4 layers; see *Layer counting*). It never sets `deep=True`.
- Non-monotone embedding: build **two** `Dense` layers at the model width —
  `embed_hidden = (width, width)` — instead of the current single `(width,)`.
  Gated by a keyword `embed_layers: int = 1` (default preserves the one-layer
  behaviour of existing callers); this run passes `embed_layers=2`. Rationale in
  *Concepts: non-monotone feature embedding*.

### Change 2 — `search.py` flavor plumbing

- `flavor_name(mode, residual, deep)` already yields `"alternate-plain"` for
  `("alternate", False, False)`; no change needed beyond allowing the mode.
- `_parse_flavors` (CLI): accept `"alternate-plain"` → `("alternate", False,
  False)`. Raise a clear error for `"alternate-residual"` / `"alternate-deep"`
  (unimplemented).
- Thread `search_activation` from `search()`'s CLI/params down into
  `suggest_config`. Default `False` preserves existing callers.
- `final_eval`: replace the hardcoded `activation="elu"` with
  `activation=best_params.get("activation", "elu")`, and set
  `alt_init="composition" if mode == "alternate" else None`. Existing studies
  (no `activation` key in `best_params`) fall back to `elu`.
- `final_eval` builds `embed_hidden = (width, width)` (two `Dense` layers) for
  this run's refit, matching the search-time embedding.

### Change 3 — `make_tables.render()`

- Add the `alternate` mode: a single `alternate | plain` row per dataset
  (alternate is plain-only).
- Recompute **bold = best per dataset** across `{split, mixed, alternate} ×
  {plain}` (residual rows are absent in this base result).
- Append a **verdict line per dataset**: the bootstrap 95% CI on the IQM delta
  `alternate − best-of-{split, mixed}` (via the existing
  `results.bootstrap_delta`), and an overall "alternate wins N of 5" summary.
- Keep the existing robustness table, extended with the per-activation breakdown
  (best activation per flavor, so the activation search is visible).

## Run matrix & budget

| axis | values |
|---|---|
| datasets | `heart`, `auto`, `compas`, `loan`, `blog` |
| flavors | `split-plain`, `mixed-plain`, `alternate-plain` |
| depth (searched) | `[1, 3]` (≤4 layers; `layers = depth + 1`) |
| activation (searched) | `relu`, `elu`, `softplus`, `selu` |
| non-monotone embedding (fixed) | two `Dense` at model width (`embed_layers=2`) |
| other HPs (searched) | width, lr, weight_decay, dropout, lr_decay, batch, `convex_fraction` (mixed only) |

Budget per `(dataset, flavor)` reuses the existing `_BUDGET`
(`n_trials`, `final_seeds`, `n_splits`) — no new budget knobs. 5 datasets × 3
flavors = **15 studies**.

### Layer counting

We count layers the way the **theorem** does. Sartor 2025 Thm 3.5 states a
non-negative-weight MLP with **3 hidden layers** interpolates any monotone
function; Prop 3.9 (the `alternate` theorem) states **≥4 layers** ⇒ universal
monotone approximator, matching Mikulincer–Reichman 2022. "3 hidden layers" and
"4 layers" describe the *same* network, so the theorem's layer count **includes
the output layer**:

`layers = (monotone hidden layers) + (monotone output layer)`.

In the builder that is the `depth` `MonoLinear` stack layers (each with an
activation) **plus** the read-out **head** (a `MonoLinear`, `identity`) — so
`layers = depth + 1`. The non-monotone feature embedding's `Dense` layers are
**not** part of the monotone approximator the theorem describes and do not
count. This is exactly the README's existing `_layers` convention for plain
(`depth + 1`), so `alternate` rows drop into the published coordinate system
with **no relabel**.

Therefore "≤4 layers" is `depth + 1 ≤ 4` ⇒ **`depth ∈ [1, 3]`** (giving 2/3/4
layers), passed as `max_depth=3`. `make_tables` keeps `layers = depth + 1` for
the plain rows.

**Why this cap is the right probe:** 4 layers (`depth = 3`) is precisely Prop
3.9's universal-approximation threshold for `alternate`. So the base result
tests `alternate` right up to — and at — the minimal depth where the theory
guarantees it is a universal monotone approximator. If `alternate` cannot beat
the incumbents by 4 layers, the deficit is not a depth-of-approximation issue.

## Resumability & execution order

- **Trial-level resume:** every study runs with `--storage-dir` so Optuna's
  SQLite study DB (`load_if_exists=True`) survives interruption; a resumed run
  continues from the last completed trial. This is already implemented in
  `search()` — the base run simply always passes a storage dir.
- **Flavor-level artifact:** each `(dataset, flavor)` writes its result JSON when
  its study + final eval complete, so a kill loses at most the in-flight
  flavor's incomplete trials (which the study DB recovers anyway).
- **Smoke first:** run the full pipeline end-to-end on `heart` (smallest, ~242
  rows) — all three flavors, search + final_eval + table row — and eyeball the
  numbers before launching `auto`, `compas`, `loan`, `blog`.
- Multi-GPU fan-out via the existing `stage2_launch.py` device pool.

## Concepts: non-monotone feature embedding

This write-up is a **deliverable** — it ships as a Concepts docs page
(`docs/concepts/non-monotone-embedding.md`, linked from the concepts index) and
is the rationale for the two-`Dense`-layer embedding above.

### The problem

A `MonoLinear` layer constrains its weights to `|W|` (non-negative effective
weights), so the network is monotone non-decreasing in *every* input it sees.
With a convex activation (e.g. ReLU) and non-negative weights it can represent
only **convex** monotone functions (Sartor 2025 Prop 3.2). But real tabular
datasets mix **monotone-constrained** features (where domain knowledge fixes a
direction) with **non-monotone** ("free") features that must be allowed an
arbitrary, unconstrained effect on the output. Feeding a free feature straight
into the monotone stack would wrongly force the output to be monotone in it.

### The construction

Free features are routed through an **unconstrained `Dense` embedding**; its
output is concatenated with the monotone-feature channels and the concatenation
is fed to the monotone stack. The network stays monotone in the *declared*
monotone inputs, while the free features reach the output only through the
unconstrained embedding — so their net effect can be arbitrary. This is the
standard embedding-composition trick the builder already implements.

### Why the embedding needs two `Dense` layers

For the free branch to represent an *arbitrary* function of the free features it
must be a universal approximator, which needs a hidden layer **plus** an output
projection — two weight layers (`Linear → act → Linear`). It is tempting to drop
the embedding's output layer and let the monotone stack's first `MonoLinear`
absorb it (two adjacent linear maps merge). That merge is invalid here: the
first `MonoLinear` is `|W|`-constrained, so it can only form **non-negative**
combinations of the embedding's hidden units, and by Prop 3.2 that is restricted
to *convex* functions of them — not a free output projection. A single `Dense`
layer therefore collapses the free branch's expressivity to convex, which is not
universal.

Giving the embedding its own unconstrained output layer (the second `Dense`)
computes the arbitrary free-feature function *before* the constrained stack,
which then reads it with a positive passthrough weight. Hence **two `Dense`
layers** (at the model width) for the non-monotone branch. (Fully-monotone
datasets have no free features, so the branch is empty and this is a no-op.)

## Metrics & verdict

- Primary metric per dataset as in the README: MSE (`auto`), RMSE (`blog`),
  accuracy (`heart`/`compas`/`loan`); direction per `_lower_is_better`.
- Report **IQM** and **mean ± std** over the final-eval seeds, plus the collapse
  flag `⚠` (seeds that degenerated), matching the existing table exactly.
- **Verdict:** for each dataset, `bootstrap_delta(alternate_values,
  best_other_values, lower_is_better=…)` gives a point estimate + 95% CI for
  `alternate − best-of-{split, mixed}`. "Alternate helps" on that dataset iff the
  CI lies strictly on the better side of zero; "matches" if it straddles zero;
  "loses" otherwise.

## Output artifacts

- `benchmarks/results/alternate-base/<dataset>-<flavor>.json` — per-study results
  (+ Optuna storage DB under a sibling `studies/` dir).
- The regenerated README-format table (from `make_tables`) with the `alternate`
  rows, bold recomputed, and the verdict summary.
- A docs page under `docs/benchmarks/` (the tuned shallow base result) and a
  README table update in the same PR.
- `docs/concepts/non-monotone-embedding.md` — the *Concepts* write-up above,
  as a docs page linked from the concepts index.

## Testing

- `suggest_config`: `mode="alternate"` yields `alt_init="composition"` and no
  `convex_fraction`; `search_activation=True` produces a config whose activation
  is one of the four; default keeps `elu` (existing tests unchanged).
- `suggest_config`: `embed_layers=2` yields `embed_hidden=(width, width)` (two
  `Dense` layers); default `embed_layers=1` keeps `(width,)` (existing tests
  unchanged). `max_depth=3` caps `depth ≤ 3`; default `max_depth=4` unchanged.
- `_parse_flavors`: `"alternate-plain"` parses; `"alternate-residual"` raises.
- `final_eval`: honors `best_params["activation"]` and sets `alt_init` for
  `alternate`.
- `make_tables`: an `alternate` record takes the bold when it is best; the
  verdict line reflects the bootstrap sign. TDD on synthetic records.
- Full backend builder coverage for `alternate` already exists
  (`test_model_builder_alternate.py`).

## Follow-ups (explicitly deferred)

1. **Residual + alternate arm** — implement `residual + alternate` (spec
   `2026-07-14-flavor-ablation-benchmark-design.md` §5.2) and add the
   `alternate-residual` row.
2. **Depth study** — the `flavor_ablation.py` depth grid, resumed from where it
   was paused, as a separate "does depth help" follow-up.
3. Reconcile the two ablation entry points (fixed-grid vs Optuna) once both
   questions are answered, to retire duplicate launcher/report code.
4. **Dropout in the monotone stack (paper fidelity).** The benchmark builder
   applies dropout only in the non-monotone `Dense` embedding branch; the
   paper's own code (`mononet/legacy/mono_dense_layer.py:395-396`,
   `create_type_1`) applies dropout *between the monotone hidden layers*. Add
   dropout between `MonoLinear` stack layers in `_build_torch_stack` + the
   jax/keras stack loops to match the paper. Safe for monotonicity (test-time
   off; train-time sub-networks stay `≥0`-weighted), but handle the interaction
   with `alternate`'s composition-aware init (dropout's `×1/(1-p)` rescaling
   perturbs the per-layer variance the init sets).
5. **HP-search sensitivity curves (best-so-far vs trial count).** Per-flavor
   optimization-history curves to check each flavor's search converged (fair
   comparison) and detect under-search — especially relevant since `compas`
   `alternate` jumped 0.706→0.730 going 25→50 trials. Reconstructable from the
   committed Optuna storage DBs (all trials stored) with no re-run; extend
   trials per flavor until the best-so-far plateaus.
