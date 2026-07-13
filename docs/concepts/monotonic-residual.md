# Deep monotonic networks with residual skips

## Motivation

Deep *plain* monotone stacks fail to train: `|W|`'s all-positive weights make layer outputs
strongly correlated, so variance compounds with depth (both `absolute` and `switch` diverge by
depth ≥ 8). Static initialization cannot fix this — a corrected per-layer init (`absolute_init`)
fixes moderate-depth trainability (depth 2–4) but cannot stabilize a genuinely deep plain stack,
because the architectural coupling remains. Residual skips address the architectural root cause.

## Construction

`MonoResidual` computes `y = g_α(α)·skip(x) + g_β(β)·F(x)`, with `sub_depth=K` making `F` a
K-deep monotone sub-stack. A deep monotone network is a uniform-width `Sequential`:

```python
from mononet.torch import MonoLinear, MonoResidual
import torch.nn as nn

W = 32
net = nn.Sequential(
    MonoLinear(n_in, W, mode="absolute", activation="elu"),
    *[MonoResidual(W, W, sub_depth=2, mode="absolute", activation="elu") for _ in range(15)],
    MonoLinear(W, 1, mode="absolute", activation="elu"),
)  # ~depth 32; uniform width => identity skips
```

`sub_depth=2` is the default, so `MonoResidual(W, W, mode="absolute", activation="elu")`
is equivalent. Uniform width means every block has `in == out` and uses pure identity skips
(the strongest warm start). Total depth ≈ `2 + n_blocks * sub_depth`.

## Requirements for skip connections and gates

Four constraints pin the design; each is load-bearing — dropping any one breaks either
monotonicity or trainability.

- **Inputs must be standardized to ≈ unit scale**, via a fixed **positive**-scale per-feature
  affine (e.g. min-max to `[0, 1]`, or division by a positive constant). This is itself a
  non-decreasing map, so it is monotonicity-preserving — unlike LayerNorm/BatchNorm, which are
  **not** safe here (a data-dependent mean subtraction and possibly-negative rescale are not
  guaranteed non-decreasing). The identity skip propagates input magnitude straight through to
  every block, and both the near-identity warm start below and the `absolute`-mode static init
  are derived assuming `x ~ O(1)`. The shipped construction is measurably sensitive to this:
  train MSE degrades from ≈`0.06` at unit scale to ≈`3.1` at `x ~ O(10)` and breaks outright
  (≈`2300`) at `x ~ O(100)` — see [Input-scale sensitivity](#input-scale-sensitivity) below for
  the full table.
- **The skip path must be monotone, near-identity at init, and gated by a strictly-positive
  `g_α` equal to `1` at init.** Identity when `in == out` (the strongest warm start); an
  `exp`-parametrized positive projection when `in != out`. This is what gives a deep stack a
  ResNet-style forward-stable warm start regardless of depth.
- **The residual path `F` must be monotone, contribute ≈ 0 at init, and — the subtle
  requirement — its weights must stay trainable at init.** Monotonicity holds by the
  `|W|`/`switch` construction for *any* weight values, so it is free. "≈ 0 at init" and "trainable
  at init" are in tension under `|W|` (see [Design choices](#design-choices-two-traps-two-fixes)
  below): meeting the first the naive way (exact-zero init) silently breaks the second. Its gate
  `g_β` must be strictly positive (monotonicity forbids a signed/ReZero-style gate) and must be
  able to **open** — move off its init value — once `F` is useful.
- **Positivity of both gates is non-negotiable.** A negative gate would flip `skip` or `F` to
  *non-increasing* and break monotonicity. Both gates are parametrized so that positivity is a
  hard invariant under free optimization — it holds at every training step, with no post-update
  projection. See [Monotonicity](#monotonicity-both-size-cases) below for the theorem and proof.

## Design choices: two traps, two fixes

`g_α` (skip gate) and `g_β` (residual gate) are the two knobs that meet the requirements above.
`g_α`'s original design is correct and unchanged. `g_β`'s original design *looked* correct but
silently failed for two independent reasons — that failure, and the fix, is the substance of this
section.

### Skip gate `g_α = elu(α) + 1` — unchanged

`g_α` must equal 1 at init (`α = 0`) so the block starts as a true identity
(`y ≈ 1·skip + g_β(0)·F(0) ≈ skip`, given `F(x) ≈ 0`). `elu(α)+1` is the natural strictly-positive
function with this property: it is `1` at `0`, smooth (C¹, including at `0`), **unbounded above**
yet only *linearly* growing for `α>0` (≈ `α+1`), and **decays to `0⁺`** as `α→−∞`. So the skip can
be freely amplified or attenuated during training without exploding. Alternatives fail a
requirement: `sigmoid` caps at 1 (skip can never amplify), `exp` grows too fast (unstable),
`softplus(0) = ln 2 ≈ 0.69` (no identity at init).

### Trap 1 — dead zone in the residual gate

At init, `F` is a random monotone sub-network — not near-identity — so engaging it typically
*raises* the loss. Gradient descent therefore pushes `β` **down**, not up. The original gate,
`scaled_elu`: `g_β = max(β,0) + ε·exp(min(β,0)/ε)` (`ε = 1e-3`), has gradient `exp(β/ε)` on the
negative side, which collapses as `β` drifts negative — at the trapped value observed in the
[trap instrumentation](#trap-instrumentation) below (`β ≈ −0.0076`), the gradient is
`≈ exp(−7.6) ≈ 5×10⁻⁴` and still shrinking. `g_β` gets pinned at `≈ ε` and stays there
indefinitely. The prior design's rationale for the exponential tail — that it "lets `F` come
online" — assumed `β` drifts *up*; a random `F` makes it drift down instead, so the tail's
escape route is on the wrong side.

**Fix: `softplus`.** `g_β = softplus(β) = ln(1+e^β)` is strictly positive, smooth, and has
gradient `σ(β) ∈ (0, 1)` **everywhere** — no dead zone on either side, so `β` can move off init in
whichever direction descent pushes it, and the gate opens. (This is a *gate* token, distinct from
the unrelated `activation="softplus"` option for `F`'s own nonlinearity.) `g_β(0) = ln 2 ≈ 0.693`
— not near 0 — which is why the original design rejected `softplus` as breaking identity-at-init.
That objection is retired by Trap 2's fix below: once `F(x) ≈ 0` at init for a different reason,
the gate's init *value* no longer matters for identity-at-init.

### Trap 2 — the `|W|` frozen-weight fixed point

Suppose Trap 1 is sidestepped the naive (Fixup) way: zero-initialize `F`'s last layer directly.
Under `absolute` mode, `F` computes with `|W|`, and `d|W|/dW` at `W = 0` is `sign(0) = 0` — an
exact **gradient fixed point**. The zeroed weights never move: `F` degenerates to a per-block
learned *constant* (only its bias moves), not an `x`-dependent function. Confirmed in the
[A-vs-B ablation](#a-vs-b-ablation) below: exact-zero init moves `0/16` blocks' last-layer
weights, and train MSE floors around `0.14`–`0.29` — better than the fully gate-trapped rows
(a constant per block still helps a little), but far from the `≈0.01` a genuinely `x`-dependent
`F` reaches.

**Fix: near-zero init, not exact-zero.** Initialize `F`'s last layer by scaling its normal-init
weight by a small factor (`near_zero_scale`, default `1e-3`) and zeroing its bias. This keeps
`F(x) ≈ 0` at init (measured init F-output RMS ≈ `0.03` at unit input scale) while keeping the
weights **nonzero**, so `sign(W) ≠ 0` and gradients flow — `F` learns genuine `x`-dependence.
Intermediate `F` layers keep normal init; only the last layer's scale sets `F`'s output magnitude
at init, so only it needs the near-zero treatment. This is Fixup's zero-init-the-branch idea,
adapted to survive the `|W|` constraint's gradient fixed point.

The scale has a narrow stable band. `1e-3` is the calibrated default (F-RMS ≈ 0.03, trains
cleanly); `near_zero_scale ≥ 1e-2` lets `F` engage too strongly at init and the deep stack blows
up the same way `off+softplus` does in the ablation below; `near_zero_scale = 0.0` reproduces the
exact-zero trap exactly. It is exposed as a **user-tunable parameter**
(`MonoResidual.near_zero_scale` / `MonoResidualConfig.near_zero_scale`, default `1e-3`) —
override it for a non-unit input regime, an unusual depth, or a different warm-start preference,
but stay inside the stable band.

### Input normalization (recap)

Neither fix above touches the input-scale requirement from the previous section: standardize
inputs via a positive per-feature affine ahead of the network, never LayerNorm/BatchNorm. See
[Input-scale sensitivity](#input-scale-sensitivity) for the measured degradation curve.

## Monotonicity (both size cases)

**Theorem.** For any parameter values, `∂yⱼ/∂xᵢ ≥ 0` for all i, j (the block is non-decreasing
in every input).

**`F` is non-decreasing** (any weights): for `absolute` mode, `h = x @ |W| + b` with
`|W| ≥ 0`, and both the convex units `act(h)` (`act' ≥ 0`) and concave units `−act(−h)`
(derivative `act'(−h)·|W| ≥ 0`) are non-decreasing in x; for `switch` mode,
`act(x @ W⁺ + b) − act(x @ W⁻ + b)` with `W⁺ = max(W,0) ≥ 0` (non-decreasing) and
`W⁻ = min(W,0) ≤ 0` (so `−act(x @ W⁻)` is non-decreasing). A `sub_depth`-deep stack of
non-decreasing maps is non-decreasing by composition.

**`skip` is non-decreasing, in both size cases:**

- **Same size (`in == out`): identity.** `skip(x) = x`, Jacobian `= I ⪰ 0` → non-decreasing.
  This is a true residual; it also provides the strongest warm start (`y ≈ x`).
- **Different size (`in ≠ out`): positive projection.** `skip(x) = x @ exp(S)`. `exp(·)` is
  elementwise `> 0`, so `∂skipⱼ/∂xᵢ = exp(Sᵢⱼ) > 0` → non-decreasing for any `S`. Storing
  the projection matrix in log-space guarantees positivity for all parameter values — this is
  the multiplicative analogue of `|W|`, and it changes dimension while preserving monotonicity
  (so the warm start is "≈ a positive projection of x", not identity).

**Combination.** Differentiating `y = g_α·skip(x) + g_β·F(x)`:

$$\frac{\partial y_j}{\partial x_i}
  = g_\alpha \frac{\partial \mathrm{skip}_j}{\partial x_i}
  + g_\beta  \frac{\partial F_j}{\partial x_i}
  \;\geq\; 0$$

since `g_α, g_β > 0` (strict, for all α, β) and both Jacobian entries are `≥ 0`: a
positive-weighted sum of non-decreasing functions is non-decreasing. ∎

**Hard invariant.** The positivity constraints are applied at call time — gates via `elu`/`exp`
(skip) or `softplus`/`scaled_elu` (residual) evaluated on unconstrained parameters; `F` via
`|W|`/`clamp`; projection via `exp` — so monotonicity holds at **every** training step without
any post-update projection. The optimizer moves α, β, W freely and the function is monotone
throughout. Monotonicity direction (±) is realized once at the network front by `MonoInput`
(sign mask); everything downstream need only be non-decreasing, which every `MonoResidual` block
is. Composition of non-decreasing maps is non-decreasing, so the whole `Sequential(...)` is
monotone. ∎ The near-zero init and `softplus` gate change *nothing* about this proof: scaled
weights and `softplus(β)` are both valid points/values of the same constraint sets.

## Forward stability, and the role of K

At init `α = β = 0 ⇒ g_α = 1`. With the current default (`softplus`, `g_β(0) = ln 2 ≈ 0.69`) the
block still starts ≈ identity, because near-zero-init `F` makes `F(x) ≈ 0` dominate the product:
`y ≈ 1·skip(x) + 0.69·(≈0) ≈ skip(x)`. (Under the pre-fix default, `scaled_elu` with a random
`F`, the block also started ≈ identity — but for the wrong reason, and it *stayed* that way; see
Trap 1 above.) Either way, a stack of blocks starts
approximately equal to the identity for uniform width, regardless of depth, so signal and
gradient propagate at approximately unit scale at init — the standard ResNet warm-start argument,
applied here to the monotone setting. This is what avoids the plain-stack blow-up that renders
depth ≥ 8 untrainable. It is a claim about **forward stability**: it says nothing about whether
`F` ends up doing anything useful once training starts (see Experiments below).

`F` itself is a **K-deep plain sub-stack** (`sub_depth = K`), which — from the `absolute`-init
analysis — blows up its own internal variance by depth ≈ 4–8, same as any plain monotone stack.
The skip re-centers only every `K` layers, so once `F` actually engages, **K must stay ≤ the
plain-blowup depth**: `K ≤ 4` keeps each `F` well-conditioned; `K = 8` lets `F` explode internally
before the next skip can help. Small `K` also means more identity-dominated blocks (each does
less work), so `K = 2` (the default) balances conditioning against per-block expressiveness. This
conditioning argument is independent of the two-traps fix — it constrains `F`'s *internal*
conditioning whenever `F` is doing real work — but it was, until this fix, untested in practice,
because pre-fix `F` never engaged enough to exercise it.

## Experiments

All numbers below are read from committed JSON under `benchmarks/results/monoresidual-gate/`
(reproduce commands inline); nothing is hand-fit to the narrative.

### Skip-K forward-stability sweep

Skip-K trainability sweep (synthetic monotone target, 300-epoch Adam; final train MSE, `<0.5` =
learns) and init input-gradient norm (conditioning). Reproduce:

```
uv run --extra torch --group bench python -m benchmarks.deep_residual_run
```

The sweep covers `mode ∈ {absolute, switch}` × `depth ∈ {4, 8, 16, 32}` × `K ∈ {plain, 1, 2, 4, 8}`
(`K > depth` is skipped, shown `—`). Final train MSE (lower is better; `1e6` = diverged / capped):

| mode | depth | plain | K=1 | K=2 | K=4 | K=8 |
|---|---|---|---|---|---|---|
| absolute | 4 | 1.75 | 0.093 | **0.090** | 0.090 | — |
| absolute | 8 | 2.00 | 0.104 | **0.101** | 0.104 | 0.172 |
| absolute | 16 | 1e6 | 0.104 | **0.103** | 0.108 | 0.721 |
| absolute | 32 | 1e6 | 0.112 | **0.111** | 0.115 | 1.108 |
| switch | 4 | 416 | 0.071 | **0.068** | 0.068 | — |
| switch | 8 | 1e6 | 0.070 | **0.070** | 0.070 | 5.455 |
| switch | 16 | 1e6 | 0.076 | **0.074** | 0.075 | 30.50 |
| switch | 32 | 1e6 | 0.089 | **0.084** | 0.087 | 26.43 |

**Reframed.** Plain stacks diverge from depth 8 (`switch`) or 16 (`absolute`); K ∈ {1, 2, 4} keep
every depth **forward-stable** (init input-gradient norm stays O(1–10), vs 1e3–1e6 for plain and
K = 8) and non-divergent, while K = 8 degrades with depth and fails outright by depth 16. This
sweep predates the two-traps fix — it was generated under the original `scaled_elu` gate with a
random (non-near-zero) `F` init, i.e. the exact pre-fix configuration instrumented in the
[trap instrumentation](#trap-instrumentation) below. **It shows non-divergence, not
depth-utilisation**: on this shallow-learnable synthetic target the skip path alone reaches
MSE ≈ 0.07–0.12 while `F` sits idle behind a trapped gate (see below) — reading "trains to
MSE ≈ 0.1" as "depth works" was the original design's error. Re-running this sweep under the A+B
fix, to see whether an engaged `F` changes these numbers, is part of the Stage-2 re-run (see
[Depth on real data](#depth-on-real-data-beforeafter) below).

### Trap instrumentation

Per-step instrumentation of the pre-fix default (`a_mode="off"` — random `F`, no near-zero init —
gate `scaled_elu`), a depth-16 `absolute` stack on a synthetic monotone teacher target. Reproduce:

```
uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_trap
```

| step | train MSE | test MSE | `g_β` | `β` | block-out RMS |
|--:|--:|--:|--:|--:|--:|
| 0 | 36.85 | 29.27 | 3.7e-4 | −0.0010 | 1.096 |
| 10 | 22.23 | 21.89 | 5e-6 | −0.0053 | 1.057 |
| 50 | 11.46 | 11.33 | 1e-6 | −0.0075 | 1.057 |
| 150 | 3.31 | 3.33 | ≈0 (<1e-6) | −0.0076 | 1.057 |
| 399 (final) | 0.89 | 0.93 | 0.000 | −0.0076 | 1.057 |

`g_β` collapses within the first ~140 steps and stays pinned at `≈0` for the rest of training,
while `β` keeps drifting further negative (deeper into the dead zone, not escaping it) and the
block-output RMS holds flat at `≈1.06` — i.e. at input scale, consistent with the skip path alone
carrying the fit. Train MSE keeps falling throughout (36.85 → 0.89) purely from the input/output
projection layers adapting around a frozen-gate residual stack; the final train MSE here
(`0.891`) matches the `off`/`scaled_elu` row of the ablation below exactly, confirming the two
benchmarks probe the same trap.

### A-vs-B ablation

Depth-16 `absolute` stack, same synthetic monotone teacher, deterministic seed. `A` = `F`'s
last-layer init (`off` = normal/random; `exactzero`; `nearzero` ×`1e-3`); `B` = residual gate
(`scaled_elu` vs `softplus`). `F-moved` = number of the 16 blocks whose `F` last-layer weights
left their init after training. Reproduce:

```
uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_ablation
```

| A (F init) | B (gate) | `g_β` | train MSE | test MSE | F-moved | verdict |
|---|---|--:|--:|--:|:-:|---|
| off | scaled_elu | 0.000 | 0.891 | 0.929 | 16/16 | gate dead-zone trap |
| exactzero | scaled_elu | 0.204 | 0.294 | 0.307 | 0/16 | F frozen → constant |
| nearzero | scaled_elu | 0.000 | 0.870 | 0.911 | 16/16 | still gate-trapped |
| off | softplus | 0.693 | 1e6 (capped) | 1e6 (capped) | 9/16 | **diverges** (random F engaged) |
| exactzero | softplus | 0.757 | 0.143 | 0.148 | 0/16 | gate opens, F still frozen |
| **nearzero** | **softplus** | **0.697–0.700** | **0.011** | **0.012** | **16/16** | **best (A+B, the fix)** |

**Reading.** The two traps are independent and need independent fixes. `softplus` (B) opens the
gate — with `scaled_elu` the gate stays shut regardless of `A` (rows 1, 3: `g_β = 0.000`
either way). Near-zero init (A) is what lets `F` learn genuine `x`-dependence — exact-zero
freezes the last-layer weights (rows 2, 5: `F-moved 0/16`, MSE floored at `0.14`–`0.29`),
near-zero trains them (row 6: `16/16`, MSE `0.011`). Neither lever alone works:
`nearzero+scaled_elu` (row 3) is still gate-trapped (MSE `0.870`); `off+softplus` (row 4)
**diverges** — a random `F`, once its gate actually opens, destabilizes the deep stack.
Only `nearzero+softplus` (row 6) both opens the gate *and* trains `F`, and by a wide margin over
every other row.

`F-moved` on the two `off` rows (1, 4) is not the same signal as on `exactzero`/`nearzero`: `F`
there starts at generic random weights, not zero, so any nonzero (even minuscule) gradient
nudges the tracked weight-norm past the movement threshold — it does not indicate `F` learned
anything useful, only that the gate wasn't a perfect `0`. The informative contrast is
`exactzero` (`0/16`, a hard `sign(0)=0` fixed point) vs `nearzero` (`16/16`).

### Input-scale sensitivity

The shipped A+B construction (depth-16, `absolute`, `nearzero`+`softplus`), with input scale
`x ~ U(0, s)` swept and the synthetic teacher target standardized as usual. Reproduce:

```
uv run --extra torch --group bench python -m benchmarks.monoresidual_gate_scale
```

| input scale `s` (`x~U(0,s)`) | init F-RMS (last block) | init block-out RMS (last block) | train MSE |
|--:|--:|--:|--:|
| 0.1 | 0.0050 | 0.173 | 0.001 |
| 1.0 | 0.0250 | 1.266 | 0.065 |
| 10.0 | 0.1465 | 10.315 | 3.126 |
| 100.0 | 1.0958 | 100.987 | 2297.1 |

The near-zero-F fix keeps each block near-identity *at init* regardless of `s` — init F-RMS
tracks `s` but stays a small fraction of the block-output RMS at every scale. But the
`absolute`-mode first layer and the near-open `softplus` gate (`g_β(0) ≈ 0.69`, not `0`) both
scale with the raw input magnitude, so the block-output RMS grows linearly with `s` and training
starts further and further from the (always unit-scale) target. Train MSE is small at `s ≤ 1`,
degrades sharply by `s = 10`, and is essentially broken by `s = 100` — the fix is necessary but
**not sufficient** without also standardizing inputs (the [requirement](#requirements-for-skip-connections-and-gates)
above).

## Why A+B

The ablation shows two *independent* traps, so it takes two *independent* fixes:

- **Trap 1 (gate dead zone) is fixed by `softplus`**, not by any change to `F`'s init — with
  `scaled_elu`, near-zero init alone still leaves `g_β` pinned at `0.000` (ablation row 3).
- **Trap 2 (`|W|` frozen-weight fixed point) is fixed by near-zero init**, not by any change to
  the gate — `softplus` alone, with `F` left at normal random init, opens the gate onto a random
  `F` and **diverges** (ablation row 4, MSE capped at `1e6`).
- **Neither fix is sufficient alone**; only the conjunction (`nearzero` + `softplus`, ablation
  row 6) both opens the gate and lets `F` train, and it does so by a wide margin over every other
  cell in the ablation.
- **Monotonicity is preserved throughout** — see [Monotonicity](#monotonicity-both-size-cases)
  above: near-zero-scaled weights and `softplus(β)` are ordinary points/values of the same
  positivity constraints that made the original design monotone, so the proof needs no
  amendment.

## Depth on real data (before/after)

Whether this now-genuinely-engaged `F` improves held-out accuracy on real datasets — versus the
shallow tuned flavors, and versus the pre-fix depth-neutral verdict — is measured in
[Deep residual accuracy](../benchmarks/deep-residual-accuracy.md) and the
[large-dataset screen](../benchmarks/large-dataset-screen.md) (#90) / synthetic depth probe (#99).

```{note}
The tables on those pages currently reflect the **pre-fix** layer (`scaled_elu` gate dead zone
starving `F` at its original random init — Trap 1), and are the depth-null results that motivated
this fix (see [Motivation](#motivation) and the trap/ablation evidence above). They will be
replaced by the Stage-2 re-run on the fixed (A+B) layer — all 10 registry datasets, both GPUs,
size-driven batch bands — per the design spec's staged plan; see PRs #90 and #99 for status.
```

## Recommendation

The shipped defaults are **A+B**: near-zero init (`near_zero_scale=1e-3`) of the default `F`'s
last layer, and the dead-zone-free `softplus` residual gate (`beta_gate="softplus"`). Both are
required — see [Why A+B](#why-ab). `scaled_elu` remains a selectable `beta_gate` token (for
reproducing pre-fix runs, not recommended for new work). The default `sub_depth=2` (a skip every
2 layers) is the sweet spot for `F`'s internal conditioning: `K ≤ 4` works, `K ≥ 8` fails, and no
normalization is needed or safe. Use `sub_depth=1` only to recover the legacy single-layer block.
Inputs must be standardized to ≈ unit scale ahead of the network (a positive per-feature affine,
never LayerNorm/BatchNorm) — the construction's warm start and static init both assume it.
