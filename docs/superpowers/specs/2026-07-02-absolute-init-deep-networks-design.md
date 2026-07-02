# Correct Static Init for the `absolute` Construction (Deep-Network Trainability) — Design

**Date:** 2026-07-02
**Status:** Approved
**Package area:** `mononet` core layers (all three backends) + `benchmarks/` diagnostics + `docs/`.
**References:** Runje & Shankaranarayana 2023 (the base `\|W\|_t` / `absolute` construction) · Sartor et al. 2025 (`switch`; §"Relaxed weight constraint → better optimization … friendlier to initialization than constraining weights to `\|W\|` directly", and "default PyTorch init, no extra tuning"). Digests under `docs/references/`.

> The `switch` construction trains from a standard init at depth; `absolute` does not,
> because constraining weights to `\|W\|` breaks the assumptions standard inits are derived
> under. We add a static, closed-form init tailored to the `absolute` transform and make it
> the default for `mode="absolute"` in all backends, then prove (test) and showcase
> (benchmark + docs) that deep `absolute` nets train.

## 1. Problem & mechanism

`absolute` computes `h = x @ |W| + b`, then splits units into convex `act(h)` and concave
`-act(-h)` (default `convex_fraction=0.5`). The current default init is `he_normal`
(Kaiming, `nonlinearity="relu"`), derived for **zero-mean** weights. Under `|W|` the weights
are half-normal (**positive mean** `E|w| = std·√(2/π)`), so:

- the linear map injects a positive mean shift that **compounds with depth** for the
  positive-mean inputs produced by monotone activations, and
- the convex/concave saturation asymmetry inflates cross-unit variance.

`switch` (`act(x@W⁺+b) − act(x@W⁻+b)`) is mean-cancelling by construction, so the same static
init trains — as the Sartor paper reports. This is an **optimization/trainability** problem
(gradient conditioning through depth), not merely forward variance; it is measured by
training deep stacks and by gradient-flow at init, not by a single forward pass.

## 2. Goals & non-goals

### Goals
- A **static, closed-form init** for `absolute` (a Kaiming-analogue for the `|W|`+convex/
  concave transform) that keeps deep stacks conditioned, made the **default** for
  `mode="absolute"` across **torch, JAX, Keras**. Explicit `InitSpec` still overrides.
- A **committed diagnostic** (the "D" characterization) comparing `switch` vs `absolute`
  trainability and gradient-flow across depth, on a synthetic monotone target.
- A **fast, deterministic CI test** proving deep `absolute` is well-conditioned at init.
- A **deep-network synthetic benchmark** (train a genuinely deep `absolute` net) with results
  **exported to a rendered docs notebook**.

### Non-goals
- No normalization layer (BN/LayerNorm) — the paper's alternative remedy is explicitly
  rejected here; this is a pure-init fix.
- No architecture/kernel change; `switch` behaviour unchanged; the stateless kernels and the
  cross-backend equivalence harness are untouched (init lives in the layer wrappers).
- No data-dependent init (LSUV) — the init stays data-free and seed-reproducible.
- The residual depth/skip-granularity ablation (separate deferred note) stays separate.

## 3. The diagnostic (D) — committed

`benchmarks/_common/init_diagnostics.py`:
- `synthetic_monotone(n, d, *, seed) -> (X, y)` — standardized features; target a known
  monotone map (e.g. `y = Σ softplus(aᵢ·xᵢ)` with `aᵢ>0`, plus small noise), so any
  depth-dependent stall is attributable to construction+init, not data difficulty.
- `grad_flow(mode, depth, *, activation, width, seed) -> dict` — build a plain `MonoLinear`
  stack, one backward pass from a synthetic batch on an **untrained** stack; return
  input-gradient norm and per-layer weight-gradient norms.
- `trainability(mode, depth, *, activation, epochs, seed) -> dict` — train the plain stack a
  fixed budget on `synthetic_monotone`; return final train loss and epochs-to-threshold.

Swept over `depth ∈ {1,2,4,8,16,32}`, `mode ∈ {switch,absolute}`, `activation ∈ {elu,relu}`,
matched width/optimizer/seed. Plain `MonoLinear` stacks only — **not** the benchmark
harness (its embedding branch and residual gates would mask the effect).

**Decision rule (pre-registered).** The `absolute` init problem is *confirmed* (→ the fix is
warranted) if, versus `switch`: fixed-budget train loss is materially worse **and the gap
widens with depth**, or init gradient norms vanish/explode by a large factor (orders of
magnitude by depth 8–16). If both modes track across the whole sweep, the fix is unnecessary
and we stop. (Given §1 we expect confirmation; this rule guards against rationalising.)

## 4. The fix — static `absolute` init (all backends)

A single backend-agnostic helper derives the init from the **activation's moments under a
standard-normal pre-activation**, parameterised by `(activation, convex_fraction)`:

`mononet/core/init.py` — `absolute_init_params(activation, convex_fraction) -> (gain, convex_bias, concave_bias)`:

- **Per-unit mean-centering.** For `H ~ N(0,1)`, a convex unit emits `act(H)` (mean `μ_a > 0`),
  a concave unit `−act(−H)` (mean `−μ_a`). Choose `δ*` with `E[act(H + δ*)] = 0`; convex units
  init bias `+δ*`, concave units `−δ*`. Every unit is then individually zero-mean, so a
  layer's output is zero-mean for **any** `convex_fraction` (including `0.0`/`1.0`) — killing
  the depth-compounding drift. `convex_fraction` enters **only** via the split point
  `c = ceil(convex_fraction · units)` (which units get `+δ*` vs `−δ*`); the bias values and
  the gain do not otherwise depend on `f`.
- **Weight scale (gain).** After centering, convex and concave units carry equal variance, so
  the gain is `f`-independent. `gain` is set so the output second moment is preserved through
  the `|W|` + centered-activation map (the Kaiming-analogue for this construction).
- **Per activation.** `(gain, δ*)` are derived from the activation's moments — analytically
  where clean, else via a fixed standard-normal **quadrature** (data-free, deterministic, so
  the init stays static and seed-reproducible). Built-ins (`elu`, `relu`, `selu`, `softplus`)
  use cached constants; the quadrature is the general fallback.

**Wiring.** Each backend's layer init (torch `_init_weight`/`MonoLinear`, JAX `nnx`, Keras
initializer) calls the shared helper and, **as the default when `mode="absolute"`**, samples
`W ~ 𝒩(0, gain²/fan_in)` and initialises the bias vector per-unit (`+δ*` for the first `c`
units, `−δ*` for the rest). The single shared helper makes the three backends init `absolute`
identically in expectation. An explicit `InitSpec`/scheme argument still overrides the whole
derivation; `switch` default init is unchanged.

**Custom / new activations.** Because `(gain, δ*)` derive from the activation's moments, a new
activation obtains a correct init automatically once its moments are computable (registered
callable → quadrature). For full manual control, the `InitSpec` override path bypasses the
derivation entirely.

The derived `(gain, δ*)` per built-in activation are **validated by the D sweep** (post-fix,
measured per-layer mean ≈ 0 and variance ratio ≈ 1 across depth); the method is fixed, the
constants are its output — not placeholders.

## 5. Fast regression test (CI)

Deterministic, cheap, per active backend (`pytest.importorskip`):
- Build a **deep** `absolute` stack (depth ~8, default init), one backward pass from a fixed
  synthetic batch; assert the **input-gradient norm sits within a bounded band** (e.g. within
  a fixed factor of 1.0 — no vanish/explode). This is the training-free primary signal.
- Optionally: a tiny deep `absolute` net drives synthetic MSE below a threshold within a few
  dozen steps (sanity that it *trains*), kept small enough for CI.
Bands/thresholds are fixed constants (from the post-fix D run), so the test is non-flaky.
A companion assertion that the **pre-fix** init would fail the band is *not* committed (we
don't keep the broken init around); the diagnostic notebook carries that comparison instead.

## 6. Deep benchmark + docs export

- A `benchmarks/` run (maintainer-run, like the flavor study) trains a genuinely **deep**
  `absolute` net (depth ~16–32) on `synthetic_monotone` and records train/val curves +
  final metric; committed under `benchmarks/results/deep-init/` (JSON; no heavy logs).
- `docs/benchmarks/deep-init.ipynb` (rendered, wired into the benchmarks toctree): renders the
  D sweep (grad-flow + train-loss vs depth, `absolute` vs `switch`, pre-fix vs post-fix) and
  the deep-net training result — the showcase that deep `absolute` now trains. Reads committed
  results; builds under strict docs (execution off).

## 7. Repo layout

```
mononet/core/init.py                # absolute_init_params(activation, convex_fraction) (shared)
mononet/{torch,jax,keras}/layers.py # apply gain + per-unit bias as default for mode="absolute"
benchmarks/_common/init_diagnostics.py   # synthetic_monotone, grad_flow, trainability
benchmarks/results/deep-init/*.json      # committed deep-net + sweep results (maintainer run)
docs/benchmarks/deep-init.ipynb          # rendered diagnostic + showcase
tests/{torch,jax,keras}/test_deep_init.py  # fast gradient-band test (importorskip)
```

## 8. Testing / CI

- CI runs the fast per-backend gradient-band test (synthetic, deterministic, no network).
- The D sweep and the deep-net benchmark are **manual maintainer runs** committed with
  results (like the flavor study); CI never runs them.
- `uv run --group bench mypy` clean; ruff; `pre-commit --all-files`; strict docs build green.
- Cross-backend: the fast test runs on the active backend; the shared gain constant guarantees
  the three backends init `absolute` identically in expectation.

## 9. Acceptance

- New default init for `mode="absolute"` in torch/JAX/Keras from a shared gain; `InitSpec`
  override intact; `switch` unchanged; kernels/equivalence untouched.
- Committed diagnostic reproduces the `absolute`-vs-`switch` depth comparison and, **post-fix**,
  shows `absolute` tracking `switch` in gradient-flow and trainability to depth 32.
- Fast CI test passes (deep `absolute` gradient-norm band) on the active backend.
- Deep synthetic `absolute` net trains; result rendered in `deep-init.ipynb`.
- All gates green; no wheel/package-contract regressions.

## 10. Open items

- For each built-in activation, whether `(gain, δ*)` is derived closed-form or by cached
  quadrature — an implementation choice; either way validated against the D sweep (per-layer
  mean ≈ 0, variance ratio ≈ 1 across depth). Per-activation differences are handled by the
  moment-based derivation itself (§4), not left open.
- Whether the quadrature grid / sample count for the general fallback needs tuning for
  accuracy vs speed — set to a fixed, documented value; validated by D.
- Deep-benchmark budget (depth, epochs, seeds) — set from the D runtime once known.
