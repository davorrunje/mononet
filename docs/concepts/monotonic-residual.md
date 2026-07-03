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

## Theory

### Why the gates are shaped this way

Three requirements pin the design, and each gate is the minimal function that meets them:

**Both gates must be strictly positive**, for every value of their unconstrained parameter.
A negative gate would flip `skip` or `F` to *non-increasing* and break monotonicity — so the
gate values, not just the layer weights, are what make monotonicity a hard invariant under
free optimization. Both `g_α = elu(α)+1` and `g_β = max(β,0) + ε·exp(min(β,0)/ε)` (ε=1e-3)
map `ℝ → (0, ∞)`.

**The skip gate `g_α` must equal 1 at init** (`α=0`) so the block starts as a true identity
(`y ≈ 1·skip + ε·F ≈ skip`). `elu(α)+1` is the natural strictly-positive function with this
property: it is `1` at `0`, smooth (C¹, including at `0`), **unbounded above** yet only
*linearly* growing for `α>0` (≈ `α+1`), and **decays to `0⁺`** as `α→−∞`. So the skip can be
freely amplified or attenuated during training without exploding. Alternatives fail a
requirement: `sigmoid` caps at 1 (skip can never amplify), `exp` grows too fast (unstable),
`softplus(0) = ln 2 ≈ 0.69` (no identity at init).

**The residual gate `g_β` must be ≈ 0 at init** so `F` starts nearly off — this is precisely
what makes deep stacks trainable (a stack of blocks ≈ identity, avoiding the plain-stack
blow-up). The `scaled_elu` form gives `g_β = ε = 1e-3` at `β=0`: tiny but **nonzero**. The
nonzero part is deliberate — a plain `ReLU(β)` would give exactly `0` at init *and zero
gradient* for `β≤0`, a **dead gate**: `F` could never learn to turn on. The `ε·exp(β/ε)` tail
keeps `g_β` strictly positive with a small but nonzero gradient near the near-zero init, so `β`
can escape `0` and `F` can come online. For `β>0` the gate is exactly linear (`= β + ε`, unbounded,
gradient 1), letting `F` grow as strong as needed without exponential instability. `ε` sets both
the init value and the width of the smooth soft-zero region.

Together: at init `y ≈ 1·skip + 1e-3·F ≈ identity` (deep-stack-friendly), and training can
*independently* scale the residual up (`β↑`) and adjust the skip (`α`), while positivity of both
gates preserves monotonicity at every step.

### Monotonicity (both size cases)

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
evaluated on unconstrained parameters; `F` via `|W|`/`clamp`; projection via `exp` — so
monotonicity holds at **every** training step without any post-update projection. The optimizer
moves α, β, W freely and the function is monotone throughout. Monotonicity direction (±) is
realized once at the network front by `MonoInput` (sign mask); everything downstream need only
be non-decreasing, which every `MonoResidual` block is. Composition of non-decreasing maps is
non-decreasing, so the whole `Sequential(...)` is monotone. ∎

### Why depth becomes trainable, and the role of K

At init `α = β = 0 ⇒ g_α = 1, g_β ≈ 1e-3 ⇒ y ≈ skip(x)`. A stack of blocks starts
approximately equal to the identity (for uniform width) regardless of depth, so signal and
gradient propagate at approximately unit scale — the standard ResNet warm-start argument,
applied here to the monotone setting. This avoids the plain-stack blow-up that renders depth
≥ 8 untrainable.

The residual branch `F` is a **K-deep plain sub-stack**, which from the absolute-init analysis
blows up its variance by depth approximately 4–8. The skip re-centers only every K layers.
Hence **K must be ≤ the plain-blowup depth**: K ≤ 4 keeps each `F` well-conditioned; K = 8
lets `F` explode internally before the skip can help. Small K also means more
identity-dominated blocks (each does less work), so **K = 2 balances conditioning against
per-block expressiveness**: it keeps each sub-stack short enough to stay well-conditioned while
giving each block enough capacity to contribute.

This is confirmed empirically: K ∈ {1, 2, 4} all train depth-32 networks to MSE ≈ 0.08–0.11
(vs plain-stack MSE ≈ 1e6); K = 8 fails (MSE > 0.5). K = 2 shows the best consistency and
margin across depths and modes.

## Experiments

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

Plain stacks diverge from depth 8 (`switch`) or 16 (`absolute`); K ∈ {1, 2, 4} train every
depth to MSE ≈ 0.07–0.12, while K = 8 degrades with depth and fails outright by depth 16. The
init input-gradient norm (`init_grad_norm` in the JSON) tracks this: it stays O(1–10) for the
trainable K and explodes to 1e3–1e6 for plain and K = 8. `sub_depth=2` (bold) gives the lowest
MSE and the tightest spread across all eight (mode, depth) cells.

## Real-dataset accuracy (forthcoming)

Stage 2 will report whether the now-trainable depth improves test metrics on real datasets vs
the shallow tuned flavors. *(Results to be added.)*

## Recommendation

The default `sub_depth=2` (a skip every 2 layers) is the sweet spot: K ≤ 4 works, K ≥ 8 fails,
and no normalization is needed. Use `sub_depth=1` only to recover the legacy single-layer block.
