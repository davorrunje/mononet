# Deep Monotonic Networks via Residual Skips (`MonoResidual.sub_depth`) — Design

**Date:** 2026-07-03
**Status:** Approved (design)
**Sub-project:** B (follow-up to A, `2026-07-02-absolute-init-deep-networks-design.md`).
**References:** Runje & Shankaranarayana 2023 (base `|W|` construction); Sartor et al. 2025 (`switch`);
He et al. 2015 (residual learning) — the near-identity-skip mechanism this adapts.

> **Goal.** Make genuinely deep (≥ 8, target 32) monotonic networks *trainable* by inserting
> near-identity monotone **residual skips every K layers**, and ship it as a thin, posture-
> respecting layer convenience (`MonoResidual` gains `sub_depth`), with **paper-grade docs**
> (the monotonicity proof + the trainability/conditioning evidence). Staged (plan C): prove
> trainability on a synthetic monotone target first; then validate downstream accuracy on real
> datasets as a follow-on.

## 1. Motivation (from A)

A established that the corrected static `absolute` init fixes *moderate*-depth trainability
(depth 2–4) but **cannot** make a genuinely deep *plain* stack forward-stable: `|W|`'s all-
positive weights make layer outputs strongly correlated (≈ 0.8), so variance compounds with
depth for **both** `absolute` and `switch`. Plain stacks diverge by depth ≥ 8; a static per-
layer init cannot fix this. B addresses it with architecture (residual skips), not init.

## 2. Empirical result that motivates the design (already run)

Skip-K trainability sweep (`mode` ∈ {absolute, switch}, depth ∈ {4,8,16,32}, K = layers per
residual block; synthetic monotone target; 300-epoch Adam; final train MSE, `<0.5` = learns):

| | plain | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|
| absolute d16 | **1e6** | 0.108 | 0.107 | 0.109 | 0.815 |
| absolute d32 | **1e6** | 0.099 | **0.101** | 0.112 | 0.509 |
| switch d16 | **1e6** | 0.074 | 0.070 | 0.067 | 31.8 |
| switch d32 | **1e6** | 0.084 | **0.075** | 0.090 | 17.9 |

Init input-gradient norm: plain explodes with depth (absolute d32 ≈ `8.6e15`); K≤4 stays
single-to-double-digit at all depths (absolute K=2 d32 ≈ 3.3); K=8 blows up (`1e4`+).

**Findings.** (1) Residual skips **fully fix** deep trainability for K ≤ 4 (depth-32 trains
to MSE ≈ 0.08–0.10 vs plain `1e6`), for both modes. (2) **K ∈ {1,2,4} all work; K=8 fails** —
each K-deep sub-stack blows up before the skip re-centers (K must be ≤ the plain-blowup depth
~4–8). (3) **K = 2 is the sweet spot** (best consistency + margin across depths, both modes).
(4) **Normalization is unnecessary** — the near-identity skip alone suffices.

## 3. Theory (must appear in the docs, paper-grade)

### 3.1 The block

A `MonoResidual` block computes
```
y = g_α(α)·skip(x)  +  g_β(β)·F(x)
```
with unconstrained scalars `α, β`, a monotone sub-module `F`, and **strictly positive gates**
`g_α = elu(α)+1 ∈ (0,∞)` (=1 at α=0) and `g_β = max(β,0) + ε·exp(min(β,0)/ε)` (ε=1e-3, ∈(0,∞),
=ε at β=0). Positivity holds for *all* real α, β.

### 3.1.1 Why these gate parametrizations

Three requirements pin the design, and each gate is the minimal function that meets them:

- **Both gates must be strictly positive**, for every value of their unconstrained parameter.
  A negative gate would flip `skip` or `F` to *non-increasing* and break monotonicity — so the
  gate values, not just the layer weights, are what make monotonicity a hard invariant under
  free optimization. Both forms map `ℝ → (0, ∞)`.

- **The skip gate `g_α` must equal 1 at init** (`α=0`) so the block starts as a *true identity*
  (`y ≈ 1·skip + ε·F ≈ skip`). `elu(α)+1` is the natural strictly-positive function with this
  property: it is `1` at `0`, smooth (C¹, including at `0`), **unbounded above** yet only
  *linearly* growing for `α>0` (`≈ α+1`), and **decays to `0⁺`** as `α→−∞`. So the skip can be
  freely amplified *or* attenuated during training without exploding. Alternatives fail a
  requirement: `sigmoid` caps at 1 (skip can never amplify), `exp` grows too fast (unstable),
  `softplus(0)=ln2≈0.69` (no identity at init).

- **The residual gate `g_β` must be ≈ 0 at init** so `F` starts *nearly off* — this is precisely
  what makes deep stacks trainable (a stack of blocks ≈ identity, avoiding the plain-stack
  blow-up). `scaled_elu` gives `g_β = ε = 1e-3` at `β=0`: tiny but **nonzero**. The nonzero part
  is deliberate — a plain `ReLU(β)` would give exactly `0` at init *and zero gradient* for `β≤0`,
  a **dead gate**: `F` could never learn to turn on. The `ε·exp(β/ε)` tail keeps `g_β` strictly
  positive with a small but **nonzero gradient near the near-zero init**, so `β` can escape `0`
  and `F` can come online. For `β>0` the gate is exactly linear (`= β + ε`, unbounded, gradient 1),
  letting `F` grow as strong as needed without exponential instability. `ε` sets both the init
  value and the width of the smooth soft-zero region.

Together: at init `y ≈ 1·skip + 1e-3·F ≈ identity` (deep-stack-friendly), and training can
*independently* scale the residual up (`β↑`) and adjust the skip (`α`), while positivity of both
gates preserves monotonicity at every step.

### 3.2 Monotonicity theorem (both size cases)

**Claim.** For any parameter values, `∂yⱼ/∂xᵢ ≥ 0` for all i, j (the block is non-decreasing
in every input).

**`F` is non-decreasing** (any weights): *absolute* — `h = x@|W|+b` with `|W| ≥ 0`, convex
units `act(h)` (`act' ≥ 0`) and concave units `−act(−h)` (derivative `act'(−h)·|W| ≥ 0`) are
both ↑; *switch* — `act(x@W⁺+b) − act(x@W⁻+b)` with `W⁺=max(W,0)≥0` (↑) and `W⁻=min(W,0)≤0`
(so `−act(x@W⁻)` ↑). A `sub_depth`-deep stack of these is a composition of non-decreasing maps
→ non-decreasing.

**`skip` is non-decreasing, in both cases:**
- **Same size (`in==out`): identity.** `skip(x)=x`, Jacobian `= I ⪰ 0` → ↑. A true residual;
  strongest warm start (`y ≈ x`).
- **Different size (`in≠out`): positive projection.** `skip(x) = x @ exp(S)`. `exp(·)` is
  elementwise `> 0`, so `∂skipⱼ/∂xᵢ = exp(Sᵢⱼ) > 0` → ↑. Storing the projection in log-space
  (`exp`) *guarantees* positivity for any `S` (the multiplicative analogue of `|W|`); it changes
  dimension while staying monotone (so the warm start is "≈ a positive projection of x", not
  identity).

**Combination.** `∂yⱼ/∂xᵢ = g_α·∂skipⱼ/∂xᵢ + g_β·∂Fⱼ/∂xᵢ ≥ 0` since `g_α,g_β > 0` and both
Jacobians are `≥ 0`: a positive-weighted sum of non-decreasing functions is non-decreasing. ∎

**Hard invariant.** The constraints are applied at call time (gates via `elu`/`exp`; `F` via
`|W|`/`clamp`; projection via `exp`), so monotonicity holds at **every** training step — the
optimizer moves α, β, W freely and the function is monotone throughout. Direction (±) is
realized once at the front by `MonoInput` (sign mask); everything after need only be non-
decreasing, which every block is. Composition ⇒ the whole `Sequential(...)` is monotone.

### 3.3 Why depth becomes trainable, and the role of K

At init `α=β=0 ⇒ g_α=1, g_β≈1e-3 ⇒ y ≈ skip(x)` — a stack of blocks starts ≈ identity
(uniform width) regardless of depth, so signal and gradient propagate at ~unit scale (the
ResNet argument), avoiding the plain-stack blow-up. The residual branch `F` is a **K-deep plain
sub-stack**, which from A blows up its variance by depth ~4–8; the skip re-centers only every K
layers. Hence **K must be ≤ the plain-blowup depth**: K ≤ 4 keeps each `F` well-conditioned
(confirmed §2); K=8 lets `F` explode internally before the skip helps. Small K also means more
identity-dominated blocks (each does less) — so **K=2 balances conditioning against per-block
expressiveness**, matching the sweep.

## 4. API — `MonoResidual.sub_depth` (all three backends)

Extend the existing `MonoResidual` layer (no new composed model class — respects the repo's
"package ships layers, users stack with native `Sequential`" posture):

- New keyword **`sub_depth: int | None = None`**, resolving to a **default of 2** — a skip every
  two layers, the §2 sweet spot. The sentinel `None` distinguishes "caller didn't set it"
  (→ 2) from an explicit value, which matters for the `F` rule below. **This changes the default
  `MonoResidual` block** (its default `F` is now a 2-layer stack, not a single `MonoLinear`);
  acceptable pre-1.0, and it makes the good deep default the *out-of-the-box* behaviour.
- When `F is None`, let `k = 2 if sub_depth is None else sub_depth`. For `k == 1`, `F` is a
  single `MonoLinear(in_features, units)` (byte-equivalent to the *legacy* default `F`); for
  `k > 1`, `F = MonoLinear(in_features, units) → MonoLinear(units, units) × (k-1)` via the
  framework `Sequential` (`MonoDense` for Keras), all sharing `mode`/`activation`/`init`.
- **`F` + an explicitly-set `sub_depth`** (`F is not None and sub_depth is not None`) →
  `ValueError` (a custom `F` owns its structure). Passing `F` **alone** is fine — `F` is used and
  the `sub_depth` default is ignored (the sentinel is what avoids a false conflict here).
- `sub_depth is not None and sub_depth < 1` → `ValueError`.
- Everything else unchanged: dual gates, near-identity warm start, identity skip when
  `in==out`, `exp`-projection skip when `in≠out`.

**Deep-network usage (documented recipe, no builder):** uniform body width `W`, residual body
sandwiched between plain projections:
```python
Sequential(
    MonoLinear(in, W, mode=..., activation=...),          # plain input projection
    *[MonoResidual(W, W, sub_depth=2, mode=..., activation=...) for _ in range(n_blocks)],
    MonoLinear(W, 1, mode=..., activation=...),           # plain output projection
)
```
(`sub_depth=2` shown for clarity, but it is now the **default**, so `MonoResidual(W, W, mode=…,
activation=…)` suffices.) Uniform `W` ⇒ every block is `in==out` ⇒ pure identity skips
(strongest warm start); depth ≈ `2 + n_blocks·sub_depth`. The default `sub_depth=2` is the §2
sweet spot; K ≤ 4 works, K ≥ 8 fails.

## 5. Components / repo layout

```
mononet/{torch,jax,keras}/layers.py   # MonoResidual gains sub_depth (default F = sub_depth-stack)
tests/{torch,jax,keras}/test_mono_residual_subdepth.py  # sub_depth builds K layers; sub_depth=1 unchanged; F+sub_depth conflict raises
tests/{torch,jax,keras}/…             # monotonicity property test (output non-decreasing in every input)
benchmarks/_common/init_diagnostics.py    # extend: build_residual_stack(mode, depth, sub_depth) helper for the sweep
benchmarks/deep_residual_run.py            # committed skip-K sweep runner -> results/deep-residual/*.json
benchmarks/results/deep-residual/*.json    # committed sweep (trainability + init grad-norm), .gitignore *.db/*.jsonl
docs/concepts/monotonic-residual.md        # PAPER-GRADE: motivation + §3 theory/proof + §2 methods & results + recommendation
tests/torch/test_deep_residual.py          # fast regression: depth-32 sub_depth=2 absolute trains (MSE < 0.5); plain diverges
```

## 6. Testing / CI

- **Monotonicity property test** (per backend, `importorskip`): for a `MonoResidual` (and a
  small `sub_depth=2` deep stack), perturbing any single input upward never decreases any
  output component (numeric check over random inputs + random parameter draws) — the paper's
  core guarantee, tested for both size cases (`in==out` and `in≠out`).
- **`sub_depth` unit tests** (per backend): `sub_depth=K` ⇒ default `F` contains `K` monotone
  layers; `sub_depth=1` byte-equivalent to current default; `F` + `sub_depth>1` raises; `sub_depth<1` raises.
- **Fast trainability regression** (`tests/torch/test_deep_residual.py`): a depth-32,
  `sub_depth=2`, `absolute` stack trains below a fixed MSE threshold in a small budget (with
  margin from §2 ≈ 0.10), and a plain depth-32 stack does not — pins the deep-trainability win.
- **Cross-backend parity**: a `sub_depth=2` `MonoResidual` given identical weights produces
  equal outputs across backends (extends the equivalence posture to the composed block; the
  stateless kernels are unchanged).
- Sweep + real-dataset accuracy are **manual/controller runs** committed with results; CI never
  runs them. `uv run --group bench mypy`, ruff, `pre-commit --all-files`, strict docs build all green.

## 7. Docs (paper-oriented)

`docs/concepts/monotonic-residual.md` (wired into the toctree), written so it can seed the
paper's architecture section and accept more experiments later:
1. **Motivation** — the deep plain-stack failure (A), with the numbers.
2. **Construction** — the block, the gates, `sub_depth`, uniform-width deep stacking.
3. **Theory** — the §3 monotonicity theorem *with proof* (both size cases), the hard-invariant
   argument, and the near-identity warm-start / K analysis. This is the section flagged as
   essential.
4. **Experiments** — the skip-K sweep (§2 table + init grad-norm conditioning), reproduced from
   committed `benchmarks/results/deep-residual/*.json`; a "reproduce" command; a placeholder
   subsection for the forthcoming real-dataset accuracy results.
5. **Recommendation** — `sub_depth=2`, K ≤ 4, no normalization needed.

## 8. Staged plan (C)

- **Stage 1 (this spec):** `sub_depth` API + tests + committed sweep + paper-grade docs.
  Success = depth-32 monotone nets train (MSE ≈ 0.1) via `sub_depth=2`; monotonicity tests pass.
- **Stage 2 (follow-on, after Stage 1 lands):** does the now-trainable depth *help accuracy* on
  1–2 real datasets vs the shallow tuned flavors (reusing the Phase-2a harness)? Documented as a
  new experiments subsection; likely more paper experiments beyond it.

## 9. Non-goals

- No normalization layer (empirically unnecessary here).
- No new composed model class / `deep_mono_stack` builder (posture: `sub_depth` on the existing
  layer + a documented `Sequential` recipe).
- No change to `MonoLinear`/`MonoInput`/kernels or the `switch`/`absolute` math.
- Not the full 5-dataset accuracy study — Stage 2 samples 1–2 datasets; broader runs are paper
  follow-on.
- Not re-running Phase-2a: the new `sub_depth=2` default flows into the benchmark harness's
  "residual" flavor (now 2-deep), so the committed `flavor-comparison` residual numbers become
  stale — re-running them is a tracked follow-up outside this sub-project (decision to accept
  this, rather than pin `sub_depth=1` in `model_builder`, was explicit).

## 10. Open items

- `sub_depth` default is **2** (sentinel `None` → 2), the §2 sweet spot — chosen so the good
  deep behaviour is out-of-the-box. This **changes the default `MonoResidual`** (default `F` is
  now a 2-layer stack); `sub_depth=1` recovers the legacy single-`MonoLinear` block. Existing
  MonoResidual tests must still pass (warm start is depth-independent); update any that assume a
  single-layer default `F`.
- Cross-backend parity test tolerance for the composed block (float32/64) — set from a first run.
- Stage-2 dataset choice (e.g. `auto` + one larger) and depth — decided when Stage 1 lands.
