# Design: Getting-started quickstart + consolidated tabbed guide

**Date:** 2026-07-12
**Status:** Approved (brainstorming) — pending implementation plan
**Scope:** Reader-facing docs. Workstream A of the documentation-audit follow-ups.

## Problem

Two findings from the [documentation audit](../audits/2026-07-12-docs-audit.md):

- **#2 (landing quickstart + routing):** `docs/index.md` is only an install
  snippet, a BibTeX block, and a hidden `toctree`. A new adopter lands with no
  "first monotonic model" and no obvious next step.
- **Guide duplication (raised during design):** `docs/guides/{pytorch,jax,keras}.md`
  are ~80% identical prose (Public API bullets, "layers only / no `MonoMLP`"
  line, the mixed-feature example preamble, "See also"). Only four things differ
  per backend. The triplication is exactly the drift risk the audit flags.

Finding **#3 (upgrade note)** is **deferred**: mononet is pre-first-release (the
whole `CHANGELOG.md` is under `[Unreleased]`, version `0.0.0a0`), so there is no
released version to migrate from yet. It becomes release-time work, naturally
grouped with the `releasing.md` touch in Workstream C.

## Goals

- A minimal, runnable, backend-tabbed **quickstart** on the landing page.
- A short **persona-routing** block so a landing adopter knows where to go.
- **One consolidated guide** with shared prose written once and only the
  backend-specific bits in tabs.

## Non-goals

- No upgrade/migration note (deferred, see above).
- No change to the `risk_net_*` examples or their tests (only their
  presentation moves into tabs).
- No IA changes beyond the guides (e.g. moving `releasing.md` is Workstream C's
  finding #13, not this).

## Files

- Create: `docs/examples/quickstart_torch.py`, `docs/examples/quickstart_jax.py`,
  `docs/examples/quickstart_keras.py` — minimal (~10-line) "first monotonic
  model" per backend.
- Create: `tests/examples/test_quickstart.py` — a smoke test per backend
  (import + forward pass, `importorskip` guard, `KERAS_BACKEND=jax`).
- Modify: `docs/index.md` — add `## Quickstart` (backend tabs) and
  `## Where to next`, placed after `## Install` and before `## Citation`.
- Modify: `docs/guides/index.md` — becomes the single guide (shared prose +
  a `{tab-set}`); drop its child `toctree`.
- Delete: `docs/guides/pytorch.md`, `docs/guides/jax.md`, `docs/guides/keras.md`.

## Landing quickstart

A `## Quickstart` section on `docs/index.md` immediately after `## Install`,
framed as "your first monotonic model." One `sphinx-design` `{tab-set}` with
three `{tab-item}`s — **PyTorch**, **JAX**, **Keras 3** — each `literalinclude`-ing
its minimal example module.

**Minimal example (same shape in all three backends, ~10 lines):** a small
fully-monotone regressor — a native `Sequential` of a dense monotone layer, a
`MonoResidual`, and a final dense monotone layer, **non-decreasing in every
input** (no `MonoInput`, to stay minimal) — followed by a forward pass on a toy
batch that prints the output shape. Uses `activation="elu"` and the default
`mode="absolute"` so a copy-paste runs. Backend specifics:

- torch: `MonoLinear(in, out, activation="elu")` + `nn.Sequential`.
- jax: `MonoLinear(in, out, activation="elu", rngs=nnx.Rngs(0))` + `nnx.Sequential`.
- keras: `MonoDense(units, activation="elu")` (build-time input width) +
  `keras.Sequential`.

Below the tabs, one line: *"The same layers exist in all three backends — see
the [guide](guides/index.md) for the full mixed-feature example."*

**Why `literalinclude` + smoke test (not an inline block, not a parity test):**
`literalinclude` pulls the file into the page at build time, so the doc cannot
drift from the source; the smoke test guards that the source itself still runs.
This is stronger and simpler than the README's hand-copied-block + parity-test
approach.

### `tests/examples/test_quickstart.py`

One test per backend, mirroring the existing `test_risk_net_*` guard style:
`torch = pytest.importorskip("torch")` (and the jax/keras equivalents,
`os.environ.setdefault("KERAS_BACKEND", "jax")` for keras), import the quickstart
module, run its model on a toy input, assert the output shape. No parity
assertion (the `literalinclude` makes drift impossible).

## Consolidated guide

`docs/guides/index.md` becomes the single guide (title stays `# Guides`).

**Shared prose, written once (above the tabs):**

- Intro paragraph: mononet ships monotonic *layers*, not composed models — stack
  them with the framework's native `Sequential`; there is no `MonoMLP`.
- **Public API** — the three-bullet list (dense layer / `MonoResidual` /
  `MonoInput`) written generically, with the backend-specific class name and
  xref deferred into the tabs.
- Mixed-feature **example preamble** (the RiskNet description), identical across
  today's three guides.
- **See also** — [Concepts: monotonicity](../concepts/monotonicity.md),
  [Layer reference](../concepts/layers.md), [Benchmarks](../benchmarks/index.md).

**`{tab-set}` — PyTorch / JAX / Keras 3 — each tab holds only what differs:**

- Intro line (torch `nn.Module`/Lightning; jax Flax NNX + `jit`/`grad`; keras
  `keras.ops`/multi-backend + `KERAS_BACKEND=jax`).
- Install extra (`[torch]` / `[jax]` / `[keras]`).
- Dense-layer name + `{py:class}` xrefs into that backend's `layers` module
  (`MonoLinear` for torch/jax, `MonoDense` for keras).
- The `risk_net_{backend}.py` `literalinclude` + trailing note (jax: explicit
  `rngs`; keras: build-time width + `get_config`/`from_config`; torch: none
  beyond the `MonotonicityMask` pointer).

Then delete `pytorch.md` / `jax.md` / `keras.md` and remove the child `toctree`
from `guides/index.md`. The top-level `docs/index.md` toctree entry `guides/index`
and README's `docs/guides/` directory link both remain valid.

## Where-to-next routing

A `## Where to next` block on `docs/index.md` after the quickstart, before
`## Citation` — a compact persona-oriented list:

- **Build something** → [Guides](guides/index.md) (full mixed-feature example,
  per-backend specifics).
- **Understand how it stays monotone** → [Concepts](concepts/index.md).
- **See it work / reproduce results** → [Benchmarks](benchmarks/index.md).
- **API details** → [reference](reference.md).

Contributor routing (→ `about/`) is intentionally omitted — a step removed from a
landing adopter and already reachable via the About nav; adding it would dilute
the get-started focus.

**Resulting landing order:** title/tagline → `## Install` → `## Quickstart`
(tabs) → `## Where to next` → `## Citation` → hidden `toctree`.

## Validation

- `./tools/build-docs.sh` — strict `-W` build succeeds (catches broken xrefs from
  the deleted guide pages, tab-directive syntax, and orphan/toctree issues).
- `uv run pytest tests/examples` — the new quickstart smoke tests pass alongside
  the existing README + `risk_net_*` parity tests.
- `./tools/check-docs.sh` — no new broken internal links/xrefs introduced by the
  guide consolidation.

## Verification

Landing renders Install → tabbed Quickstart → Where-to-next → Citation; the
single guide renders shared prose with a working three-backend tab-set; the three
per-backend guide files are gone with no dangling references; all three gates
above are green.
