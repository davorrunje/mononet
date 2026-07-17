# Benchmark experiment orchestration

**Status:** design
**Date:** 2026-07-15
**Author:** Davor Runje
**Scope:** repo-only `benchmarks/` tooling — no impact on the `mononet` wheel.

## Problem

The `benchmarks/` folder has mature *math and training* reuse (`_common/runner.py`,
`_common/gpu_pool.fan_out`, `_common/results.py`, `_common/config.BenchmarkConfig`,
`_common/search.py`), but the **experiment-definition and orchestration layer is
reinvented per study**. Each ablation ships a bespoke triplet — `*_run.py` +
`*_launch.py` + `RUNBOOK-*.md` — plus a `_common/*_report.py` and a `results/<study>/`
directory. There are ~20 study specs and ~40 plans of this shape.

Three coupled pains follow, in the user's priority order:

1. **Selective re-execution (top priority).** When an architectural component changes,
   there is no systematic way to re-run only the *affected* experiments. Reproducibility
   degrades as the code moves under committed results.
2. **Operational robustness.** Fan-out across GPUs, resumability, and progress/ETA
   logging get bolted onto a run *after* it has started — the user interrupts running
   GPU jobs to add plumbing.
3. **Reuse.** Because every study starts from a blank `*_run.py`, Claude writes
   orchestration from scratch instead of reusing the polished version.

These are not three separate systems; they nest. A dependency graph over experiments is
the spine; how each node executes (resume / fan-out / logging) is operational robustness;
what the nodes point at (declarative experiment definitions) is reuse. Solving (1)
correctly forces the abstractions that deliver (2) and (3).

## Role as the default experiment backend

This spec is the **default, reference implementation** of the *experiment-backend contract*
consumed by the research-workflow skills (defined in
[docs/research/README.md](../../research/README.md)). It provides the contract's four
capabilities: **run** (execute a hypothesis's designed experiments — `run` / `reconcile`),
**evidence** (a committed result carrying a *run-ref* in `.runs/<run-hash>.json` and a
`.provenance.json` *provenance stamp*), **tables** (`render` managed blocks), and
**is-current** (`render --check` / provenance diff). Those skills depend on the *contract*,
not on this spec's internals — so a future paper with different execution needs could bind
a different backend without changing them (this is the default; the PINN application paper,
today a standalone draft, adopts *this* backend once the infra lands). Everything below
is this backend's binding of those capabilities.

## Compute environment

Single host: 32-core Threadripper, 256 GB RAM, **heterogeneous** NVIDIA 5090 (`cuda:0`)
+ 3090 (`cuda:1`). No cluster, no multi-node scheduler — now or within this paper's
timeline. Two consequences:

- **Work-stealing already handles GPU heterogeneity.** `gpu_pool.fan_out` dispatches
  each item to whichever device frees first, so the 5090 naturally pulls more work than
  the 3090 without weighting logic.
- **CPU is a first-class execution target.** The tiny tabular MLPs are CPU-bound and
  often *faster* on CPU (per `benchmarks/README.md`); 32 cores run ~28 CPU-slot
  experiments in parallel while the two GPUs take the large-dataset legs. The executor
  therefore drives a **mixed device pool** — `["cuda:0", "cuda:1", "cpu"×N]` — and each
  experiment declares its execution class.

## Tooling decision: `doit` core + custom symbol-hash & fan-out

Re-weighed build-vs-buy at the *layer* level (an earlier draft evaluated only Snakemake
and jumped to fully-custom). The orchestrator decomposes into layers that answer
differently:

- **Symbol-closure staleness** — the priority-#1 feature (change `MonoResidual` → rerun
  only the experiments that declared it). **No standard tool does symbol-level**; it is
  `inspect.getsource` hashing, a small custom module in *every* option.
- **DAG + up-to-date checking + task state + parallel CLI** — **adopt `doit`**. It is a
  pure-Python task runner whose defining feature is pluggable `uptodate` callables: we hand
  it our symbol-closure hash as the staleness check, generate its tasks from the TOML
  `ExperimentSpec` registry, and get the dependency graph, state persistence, `-n`
  parallelism, and a maintained CLI for free — the most error-prone machinery we would
  otherwise own and maintain. `doit` is light (no daemon, minimal deps), so it does **not**
  trip the anti-heavy-dependency posture that (correctly) ruled out Snakemake.
- **Fan-out over the heterogeneous 5090/3090/CPU pool** (device-pinning + work-stealing) —
  stays the existing `gpu_pool.fan_out`; `doit`'s built-in parallelism is not device-aware.
- **Streaming progress/ETA, `render` write-back, committed-JSON provenance** — stay custom;
  no standard tool serves the single-host heterogeneous case well, and committed JSON is
  more git-native and reproducible than an MLflow/W&B store.

Rejected alternatives: **Snakemake** — file-*mtime* staleness fights symbol-closure, heavy
`.smk` DSL. **Fully-custom** — cohesive, but we own the DAG/state/CLI `doit` gives for free.
**DVC** — recognized and git-integrated, but staleness is file-content-level, so an edit to
`layers.py` reruns everything downstream: it loses the symbol-level selectivity that is
priority #1. **Dagster / Ray** — powerful asset-versioning / resource scheduling, but heavy
platform/daemon deps, overkill for a single host, and their version field is still a manual
per-asset string we would compute ourselves.

## Architecture

New subpackage under the repo-only `benchmarks/`. Nothing here ships in the wheel.

### Code layout

```
benchmarks/_common/experiments/
├── spec.py         # ExperimentSpec (frozen dataclass) — the declarative unit
├── registry.py     # discover/load TOML specs; filter by group
├── provenance.py   # closure hashing + committed sidecar + the doit `uptodate` callable
├── executor.py     # mixed CPU/GPU work-stealing fan-out (wraps gpu_pool.fan_out)
├── tasks.py        # doit task-creators: bind specs → doit tasks (uptodate=provenance, action=executor)
└── cli.py          # Typer app (run/reconcile/status/render) wrapping doit
```

- `spec.py`, `registry.py`, `provenance.py` are pure and hardware-free — unit-testable
  on CPU with no GPU and no training.
- `tasks.py` is the **`doit` integration seam**: each experiment becomes a `doit` task
  whose `uptodate` is the symbol-closure check (`provenance`) and whose action fans the
  experiment's work across the device pool (`executor`). `doit` owns the DAG, the ordering
  (`downstream` → `task_dep`), up-to-date evaluation, and task state (`.doit.db`,
  git-ignored).
- `executor.py` is the only module that touches subprocesses/devices; it reuses
  `gpu_pool.fan_out` rather than reimplementing fan-out.
- `cli.py` uses **Typer** and wraps `doit`: `reconcile` runs the out-of-date tasks in
  scope, `status` is `doit`'s dry-run listing the stale set + prior ETA, `run` forces a
  target, `render` is independent of `doit`. (`bench` dependency group only.)
- Existing `runner.py`, `results.py`, `config.py`, `model_builder.py`, `search.py`, and
  the `*_report.py` modules are **unchanged** — the new layer orchestrates them.

Boundary test: `spec.py` tells you what an experiment *is* without revealing how it
*runs*; the executor can change without touching any spec.

### Specification files

Declarative *data*, git-tracked, separate from the orchestrator code:

```
benchmarks/experiments/
├── residual/
│   ├── group.toml                 # family-level: shared dep closure, exec class, report, downstream
│   ├── gate-ablation.toml
│   └── deep-residual-accuracy.toml
├── flavor/
│   ├── group.toml
│   └── bake-off.toml
├── depth/
│   ├── group.toml
│   └── stage2-unified.toml
└── size-ladder/
    ├── group.toml
    └── loan.toml
```

One directory per **group** (family). `group.toml` holds what the family shares — most
importantly the **declared dependency closure**, declared once. Each `<experiment>.toml`
holds one experiment's specifics and inherits the group. This realizes "declared deps are
the source of truth, groups are the convenience/addressing layer."

`residual/group.toml`:
```toml
[group]
name = "residual"
exec_class = "cpu"                  # tiny tabular MLPs → CPU slots
report = "benchmarks._common.deep_residual_report:render"
downstream = ["depth"]              # reconcile ordering: residual feeds depth analysis

deps.symbols = [                    # staleness source of truth for the whole family
  "mononet.torch.layers.MonoResidual",
  "mononet.torch._kernels.monotone_residual",
  "mononet.core.init.absolute_init_params",
]
```

`residual/gate-ablation.toml`:
```toml
[experiment]
name = "gate-ablation"
datasets = ["loan", "heart"]        # resolved via benchmarks.datasets.registry
base = "benchmarks/configs/loan.toml"   # reuse existing per-dataset config fragments
seeds = [0, 1, 2, 3, 4]

[experiment.grid]                   # cartesian product → BenchmarkConfigs
residual = [true]
depth = [2, 4, 8]
gate_init = ["absolute", "identity"]

deps.symbols = ["mononet.core.init.residual_gate_init"]   # extends the group closure
```

**Grid vs builder escape hatch.** Most ablations are grids: `registry.py` expands
`[experiment.grid]` into the cartesian product of `BenchmarkConfig`s. For anything a grid
cannot express (Optuna HP search, conditional params), the spec instead sets
`builder = "benchmarks._common.search_spaces:phase2_space"` — a dotted path to an
existing, tested Python config-builder. Either way, **adding an ablation = a new TOML,
never a new `*_run.py`.**

**Baselines / other-method comparison** are modeled as their **own experiment** in a
`comparison/` group (its builder wraps `benchmarks/baselines/`, e.g. XGBoost), producing
result JSON like any other experiment. Comparison tables reference that experiment's
results at render time — baselines are *not* embedded as a column in every study's spec.

**Why TOML, not Python specs:** matches the existing `configs/*.toml` and
`datasets/manifest.toml` conventions, keeps specs un-runnable data (fill a slot, don't
write code), and the loader validates into a frozen dataclass so most strict-mypy safety
is retained at the boundary.

## Provenance and staleness

Three hashes, distinct roles:

- **config-hash** = hash of a resolved `BenchmarkConfig`. Names one result file within a
  grid.
- **provenance-hash** = hash of `(config, declared-source, dataset-content)`. The
  **staleness key**, stored in the result's `.provenance.json` sidecar. The installed
  `mononet` version and git SHA are recorded in the sidecar for *audit* but are
  deliberately **not** part of the key — the declared-source hash is the real change
  signal, and keying on a version string would either false-positive (version bumped, no
  relevant code change) or false-negative (dev install, code changed, version static).
- **run-hash** = hash of the set of provenance-hashes in one `run`/`reconcile` invocation
  plus the git SHA. Keys the frozen per-run record.

**Declared source hashing.** Each declared symbol (dotted path) is resolved and hashed
via `inspect.getsource` of the object, so editing `MonoResidual`'s body changes only the
provenance of experiments that declared it — the flavor bake-off (which declares
`MonoLinear` only) is correctly skipped. The closure is **explicit**: if `MonoResidual`
delegates to a kernel the experiment cares about, the group must also declare that kernel.
This maintenance cost is accepted deliberately; it is the price of symbol-level precision
that file-level tools (Snakemake, Make) cannot offer. Closures are **per-backend**: an
experiment that runs more than one backend declares each backend's symbols (e.g. both
`mononet.torch.layers.MonoResidual` and `mononet.jax.layers.MonoResidual`), so a JAX-only
edit re-runs only the JAX arm. (A future lint that warns when a
declared module imports undeclared symbols is out of scope — see Follow-ups.)

**Dataset content hash** is resolved through the existing `benchmarks.datasets` registry
(LFS-backed content / manifest hash), not recomputed here.

### Two execution modes

- **Focused `run`** — execute a named spec or group *now*, ignoring the rest of the
  graph. The interactive tier: "run where I think the signal is."
- **Batch `reconcile`** — recompute provenance-hashes across the selected scope, diff
  against stored sidecars, and run exactly the stale set, ordered by `downstream` links,
  across the device pool. The overnight tier: "this change is real → bring every affected
  experiment back into a consistent state on free GPUs."

Scope for either mode is addressable by group name (`--group residual`), specific
experiment, or `--all`. Within any scope, the provenance diff decides what is actually
stale — groups address, declared deps decide.

**`downstream` is reconcile *ordering* only, for now.** Experiments train independently;
cross-experiment comparison happens at render time (a combined table reads several
experiments' current result JSON), so a changed upstream result propagates to synthesis
surfaces through `render`, not through re-training. **Deferred:** if a future experiment
genuinely consumes another's computed output (warm-start from a trained model, or reads
its result JSON as training input), the graph gains a second edge type — result-hash
propagation, where a changed upstream result marks a downstream experiment stale even
when its own code closure is unchanged. Not built now (see Follow-ups).

## Executor and resumability

**`doit` ↔ `gpu_pool` boundary.** `doit` decides *which* experiments are out-of-date (the
symbol-hash `uptodate` plus the committed-result skip) and the order (`downstream` →
`task_dep`); the *execution* of each task fans its work across the device pool via
`gpu_pool.fan_out`. Two concurrency models, chosen at implementation: (i)
**experiment-granular tasks** (default) — `doit` runs one experiment at a time and each
task saturates the pool across its configs/seeds/trials; simple, no cross-experiment device
contention; (ii) **item-granular tasks** under `doit -n` plus a shared device-lease shim —
more cross-experiment concurrency for many tiny experiments, at the cost of a small lease
coordinator. Start with (i).

`executor.py` wraps `gpu_pool.fan_out` over a **mixed pool**. Each experiment's
`exec_class` (`cpu` | `gpu`) routes its items to CPU slots or GPU devices; the
work-stealing queue balances the heterogeneous GPUs automatically. Every command it
builds is single-threaded (threaded Optuna deadlocks under process/thread nesting — an
existing hard constraint documented in `gpu_pool.py` / `stage2_launch.py`).

**Batch failure policy.** Today's `fan_out` runs items `check=True`, so the first failure
aborts the whole fan-out — wrong for an overnight `reconcile`. The executor instead
**continues past a failed item**, records the failure (with traceback) in the run record,
and exits non-zero if any item failed. One diverged or crashed experiment must not sink
the other 19.

**Dataset preflight.** Before running, the executor resolves every referenced dataset
through `benchmarks.datasets.registry` and verifies it is present (triggering
`datasets.download` if not), so an overnight batch fails fast at launch on a missing
dataset rather than hours in.

Resumability has two granularities:

| Item type | Checkpoint artifact | Resume granularity |
|---|---|---|
| Grid item — one `BenchmarkConfig` (× seeds) | committed result JSON + `.provenance.json` | item: skip if provenance-hash matches |
| Optuna study — one dataset×flavor search | Optuna SQLite `.db` (LFS-committed) | trial: completed trials skipped; study fills remaining `n_trials` |

**Item-level** — before running a grid item, `provenance.py` checks whether a result with
a matching provenance-hash exists and skips it, so a killed reconcile resumes at item
granularity.

**Trial-level** — Optuna's SQLite storage, already in use; the new layer only makes the
study path deterministic from the spec. A killed study resumes by re-invoking the same
study name against the same `.db`; completed trials are persisted and skipped, and the
study fills the remaining `n_trials`. Intra-trial (epoch) checkpointing is deliberately
**out of scope**: these trials are short and a dropped trial simply re-runs from the start
of that trial, which is cheaper than the checkpoint bookkeeping would cost.

**Experiment-level (`doit` state)** — `doit` persists each task's success and its
`uptodate` values in `.doit.db` (git-ignored), so a killed `reconcile` restarts and skips
already-completed experiments without recomputation, above the item level. The committed
`.provenance.json` sidecars remain the durable, reviewable **source of truth** for
staleness; `.doit.db` is a rebuildable local cache — deleting it costs a re-derivation of
hashes, not results.

## Progress, ETA, and storage

### Storage layout

The axis that matters is **mutating checkpoint** vs **immutable historical record**.

```
benchmarks/results/<group>/<experiment>/
├── <config-hash>.json                # result rows              — committed (plain git; ~1 KB)
├── <config-hash>.provenance.json     # deps/config/data hashes  — committed (plain git)
└── studies/<dataset>-<flavor>.db     # Optuna study             — committed via LFS

benchmarks/results/.runs/<run-hash>.json    # frozen per-run record  — committed (plain git)
benchmarks/results/.timing/ledger.jsonl     # append-only ETA index  — committed
```

- **Studies → LFS, committed.** Already the established repo convention (`.gitattributes`
  tracks `*.db filter=lfs`; `results/*/studies/*.db` are committed). Preserves the entire
  search landscape (every trial's params/value/timing), reanalyzable and resume-across-
  sessions. The new layer makes the path deterministic from the spec.
- **`.runstate.json` — live vs frozen.** During a run, a live untracked `.runstate.json`
  records per-item status (`pending`/`running`/`done`/`failed`), device, and timestamps —
  the live-progress source and crash-recovery record. Committing every mutation would
  bloat history, so on completion it is **frozen** to an immutable
  `results/.runs/<run-hash>.json` and committed. Because the key is the run-hash,
  re-running the "same" experiment under changed code lands a **new** record → a
  historical series across code versions, without churn.
- **Frozen run-record contents** (guaranteed): git SHA, per-spec provenance-hashes,
  device map, per-item durations, failures/divergences, total wall-clock. Enough to
  reproduce and to audit an overnight batch.
- **Timing ledger** is the append-only fast-path for ETA and a formal *materialized view*
  — regenerable from `.runs/*.json`. Run records are primary; the ledger is a cache.

**LFS bloat watch-item.** Committing a resumed study's `.db` at each reconcile stores a
new full LFS blob each time (SQLite is not delta-friendly). Negligible now (~14 MB total
results). If study sizes grow: commit the `.db` only on study *completion* (frozen, not
mid-resume) and periodically prune superseded versions. Documented, not solved now.

### ETA

One record per completed work item is appended to the ledger:

```json
{"exp": "residual/gate-ablation", "dataset": "loan", "device": "cuda:0",
 "n_train": 9578, "epochs": 200, "params": 4421, "n_trials": 50, "seconds": 84.2}
```

**Throughput model: one scalar per device.** Runtime is modeled as proportional to a
single composite cost:

```
cost    = n_train × epochs × params × n_trials × n_seeds
seconds ≈ cost / throughput[device]      # throughput calibrated per device from the ledger
```

Three calibrated numbers (5090 / 3090 / cpu). Deliberately not a log-linear fit: the
prior only needs to be roughly right because online refinement takes over within a few
trials/items, and a fit needs history a fresh task does not have (chicken-and-egg). Grow
to a per-device log-linear fit only if the launch-time prior is consistently off by more
than ~2×; the refinement machinery is unchanged either way.

**Three ETA regimes:**

- **Known/unchanged** (provenance seen before): ETA = ledger's recorded seconds for those
  items, summed over the fan-out and divided by parallel slots per device.
- **New/significantly-changed** (provenance-hash absent from ledger): launch with the
  cost-model prior — coarse but never zero.
- **Refinement while running** (replaces the prior as samples arrive): Optuna studies —
  `study_ETA = (n_trials − done) × trimmed_mean(recent trial seconds)`, using Optuna's
  own trial timestamps (no extra instrumentation, absorbs pruning). Grid studies —
  running mean of completed item durations.

### Live progress and resolution

Today's `fan_out` uses blocking `subprocess.run(check=True)` and captures no output, so
its progress resolution is **one event per subprocess** — a whole study or grid item.
That is too coarse (minutes of blindness during a 50-trial study). The high-resolution
signal already exists in the run; we surface it by **streaming** rather than fire-and-wait.

Mechanism: `gpu_pool` grows a streaming variant using `Popen` + a per-subprocess reader
thread. Children emit sentinel-prefixed JSONL progress events on stdout (`@@PROG@@ {...}`);
the reader routes event lines to the parent aggregator and passes all other lines through
to a per-item log. The **parent is the sole writer** of `.runstate.json`, the ledger, and
the `rich` table — no multi-writer file races. Children flush per event
(`PYTHONUNBUFFERED`).

Resolution ladder, all achievable:

- **Trial-level (backbone, ~free).** Optuna persists every completed trial to the `.db`
  with start/complete timestamps; a `callbacks=[emit]` argument to `study.optimize` emits
  one event per finished trial (`trial k/n`, value, elapsed). This is the search atom and
  drives the tightening ETA.
- **Item-level (grid).** One event per `(config, seed)` completion.
- **Epoch-level (optional, opt-in).** A *read-only* per-epoch callback in the runner loop
  — no checkpoint, no RNG state, decoupled from the trial-level resume decision. Off by
  default; enabled per experiment only for long large-dataset trials.

The parent renders a **`rich`** live table (companion to Typer, `bench` group): per-device
rows with current item, done/total, elapsed, and ETA, plus a suite-level total ETA.
**Stall detection** falls out of streaming — an item that emits no event for T seconds is
flagged rather than appearing hung. On reconnect after a crash, the table rebuilds from
`.runstate.json` + the ledger.

## CLI surface (Typer)

```
mononet-bench run       <group|group/experiment> [--all]      # focused, now
mononet-bench reconcile [--group G | --all] [--devices ...]   # batch stale set
mononet-bench status    [--group G | --all]                   # dry-run: what is stale, prior ETA
mononet-bench render    [--group G | --all] [--check]         # regenerate every quoting surface from results
```

`status` is the pre-flight: it prints the stale set and the prior ETA without running
anything — the check before kicking off an overnight `reconcile`. Under the hood `status`
is `doit`'s dry-run and `reconcile` runs `doit`'s out-of-date tasks in scope; the Typer
wrapper adds the paper-friendly verbs, scope selection, the device pool, and the ETA
table. `render` closes the loop back into docs (see next section) and is independent of
`doit`; `--check` fails without writing when any surface is stale, for use in
pre-commit/CI.

## Closing the loop: results back into docs and README

Committed result JSON is the **single source of truth for every quoted number**. No
figure is hand-typed into prose; each is generated, and a hook fails the commit if a
generated surface drifts from the results. This is the write-back half of reproducibility:
change code → `reconcile` → results change → `render` → docs/README update, with the hook
guaranteeing the last step was not skipped.

Two surface kinds, one source:

- **Docs notebooks** (`docs/benchmarks/*.ipynb`). Already the right pattern: each reads
  committed result JSON and renders its table/figure at build time; the Sphinx config
  keeps `execution_mode="off"` and outputs are pre-executed via the existing
  `tools/execute-benchmarks.sh`. `render` re-executes exactly the notebooks whose backing
  results changed.
- **Prose surfaces** (`README.md`, `benchmarks/README.md`, docs prose pages, and any
  other place that quotes a number). Each quoted table lives inside a **managed block**
  delimited by markers:

  ```markdown
  <!-- BEGIN GENERATED: flavor/bake-off -->
  ... table rewritten from results, do not edit by hand ...
  <!-- END GENERATED: flavor/bake-off -->
  ```

  `render` rewrites every managed block from the fragment its group's `report` hook
  produces (the hook already owns table rendering; it now also emits a markdown fragment
  keyed by an id such as `flavor/bake-off`). The marker id maps a fragment to every file
  that embeds it, so one result can feed README *and* a docs page without duplication.

**Staleness guard.** A pre-commit hook runs `mononet-bench render --check`: it re-derives
every managed block and notebook-backing table from current result JSON and fails if any
committed surface differs. So a changed result cannot land without its quotes being
re-rendered — the same guarantee the equivalence-hash hook already gives the reference
implementation.

**Not auto-managed:** historical design specs under `docs/superpowers/specs/` quote
point-in-time numbers as narrative and are deliberately left untouched — they record what
was true when written, consistent with the repo's memory/provenance posture.

## Documentation

Two deliverables, distinct audiences:

- **`CLAUDE.md` (contributor/agent instructions).** A concise section — the enforcement
  point for the reuse goal — stating: benchmarks are defined as TOML specs under
  `benchmarks/experiments/`; **add or edit a spec, never write a bespoke `*_run.py` /
  `*_launch.py`**; drive everything through `mononet-bench run|reconcile|status|render`;
  results are the single source of truth and quotes are regenerated, never hand-edited.
  Kept terse (CLAUDE.md is a rules index), pointing to the docs page for the full how-to.
- **Sphinx docs page** (`docs/benchmarks/orchestration.md`, linked from
  `docs/benchmarks/index`). The full user-facing guide: authoring a spec (group + grid or
  builder), declaring the dependency closure, the focused-run vs overnight-reconcile
  workflow, reading progress/ETA, and how `render` flows results back into the docs.

## Dependencies

Added to the **`bench` dependency group only** (never the wheel): `doit` (DAG / up-to-date
/ task-state / parallel CLI core), `typer` (the wrapping CLI), and `rich` (progress table).
All three are pure-Python dev-tool dependencies for the repo-only benchmark harness; none
pulls a daemon, server, or native binary.

## Migration

- Port existing studies to specs incrementally, one group at a time; a ported study's
  numbers must reproduce (within seed noise) before its old `*_run.py`/`*_launch.py`/
  `RUNBOOK-*.md` triplet is removed.
- `_common` math/training/report modules stay; specs point their `report` hook at the
  existing `*_report.py` renderers.
- Existing committed results predate provenance sidecars, so their staleness cannot be
  reconstructed. The **first `reconcile` after a group is ported re-runs it once** to
  establish baseline `.provenance.json` sidecars; thereafter selective re-execution
  applies. Where a result is known-current, its sidecar may instead be written from the
  spec without recompute — a per-group migration judgment, not an automatic guarantee.
- **Reconcile `benchmarks/README.md`** with the new storage layout in this same PR:
  correct the stale "do not commit `*.db`" line (studies are LFS-committed), and document
  `.runs/` and the timing ledger.

## Testing

- `spec.py` / `registry.py` / `provenance.py`: pure, unit-tested on CPU — grid expansion,
  group inheritance, TOML validation, and the staleness diff (edit a declared symbol's
  source → the right experiments go stale; edit an *undeclared* symbol → none do).
- `tasks.py` / `doit`: the `uptodate` wiring reports a task out-of-date iff its declared
  closure changed (editing a declared symbol → out-of-date; an undeclared one → up-to-date),
  and experiment-level resume works (`.doit.db` skips a completed task on restart, and
  rebuilds correctly after the DB is deleted since the committed sidecar is source of truth).
- `executor.py`: a small end-to-end grid on CPU slots verifying fan-out, item-level
  resume (kill + restart skips completed items), and ledger append.
- ETA: unit tests on the throughput calibration and the online-refinement update.
- `render`: idempotence (rendering twice is a no-op) and the `--check` staleness guard
  (mutating a result JSON makes `--check` fail; re-rendering makes it pass).

## Out of scope / YAGNI

- Multi-node / cluster execution (single host by decision).
- Intra-trial (epoch-level) Optuna checkpointing — trials are short; a dropped trial
  re-runs from its start (by decision).
- Log-linear ETA fit (start scalar; grow only if needed).
- A dependency lint that flags undeclared imported symbols (Follow-ups).
- Any change to the `mononet` wheel or its dependency surface.

## Follow-ups (to become GitHub issues)

- Result-hash propagation (second edge type): re-run a downstream experiment when an
  upstream experiment's *results* change, once an experiment genuinely consumes another's
  computed output. Deferred until that need arises.
- Optional closure lint: warn when a declared module imports symbols absent from the
  experiment's declared closure, to catch silent under-declaration.
- Revisit the throughput model (scalar → log-linear) if launch-time ETA is consistently
  off by > ~2×.
- LFS bloat mitigation (freeze-on-complete `.db` commit + prune) if study sizes grow.
