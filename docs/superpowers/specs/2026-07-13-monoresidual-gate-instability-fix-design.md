# MonoResidual Gate Instability Fix (Identity-at-Init F + Dead-Zone-Free Gate) — Design

**Date:** 2026-07-13
**Status:** Draft (design)
**Sub-project:** B (amends `2026-07-03-deep-monotonic-residual-design.md` §3.1.1 / §3.3).
**Package area:** `mononet` core (`reference.py`, `config.py`) + all three backends
(`{torch,jax,keras}/{_kernels,layers}.py`) + equivalence harness + `benchmarks/` + `docs/`.
**References:** Runje & Shankaranarayana 2023 (base `|W|`); Sartor et al. 2025 (`switch`);
He et al. 2015 (residual learning); Bachlechner et al. 2020 (ReZero — signed-α identity warm
start, *unavailable here* because monotonicity forbids a signed gate); Zhang et al. 2019 (Fixup
— zero-init of the residual branch's last layer, which is fix **A** below).

> **Goal.** Make deep `MonoResidual` stacks actually **use their depth**. The current design is
> monotone and non-divergent at depth, but its F-path gate `g_β` is trapped near zero from init,
> so the residual branch `F` never engages and the "deep" stack is effectively shallow — the root
> cause of the depth-null results (#90, #99). Fix = **(A)** identity-at-init `F` (zero-init its
> last layer) + **(B)** a dead-zone-free positive gate (`softplus`), shipped as the new default
> for `MonoResidual` across all three backends, with **paper-grade docs justifying the gate
> design, backed by the trap instrumentation + the A-vs-B ablation + before/after re-runs**.

## 1. Problem: the residual-gate bootstrap trap

`MonoResidual` computes `y = g_α(α)·skip(x) + g_β(β)·F(x)` with strictly-positive gates
(monotonicity requires `g_α, g_β ≥ 0`; a signed gate would flip a branch to non-increasing).
The F-path gate token is `scaled_elu`: `g_β = max(β,0) + ε·exp(min(β,0)/ε)`, `ε=1e-3`, so
`g_β = ε` at `β=0`.

The approved 2026-07-03 spec (§3.1.1) claims the `ε·exp(β/ε)` tail gives "*a small but nonzero
gradient near the near-zero init, so β can escape 0 and F can come online.*" **This does not
happen.** The claim tacitly assumes `β` drifts *up* (that engaging `F` helps). But at init `F`
is **random**, so engaging it *raises* loss ⇒ `∂loss/∂β > 0` ⇒ gradient descent drives `β`
**down**. On the negative side `scaled_elu`'s gradient is `exp(β/ε)`, which collapses almost
immediately (at the empirically observed trapped value `β ≈ −0.0076`, gradient `≈ e^{−7.6} ≈
5e-4`). `β` is pinned negative, `g_β ≈ 0` **forever**, `F` never engages, and a depth-`D` stack
is really `MonoLinear(in→W) → (scaled identity)×D → head` — a shallow net.

**Why this was invisible.** The 2026-07-03 §2 sweep reported depth-32 stacks training to
MSE ≈ 0.08–0.10 and concluded "residual skips fully fix deep trainability." That measurement is
correct but answers the wrong question: on a shallow-learnable monotone target the stack trains
fine **via the skip path** with `F` off. It demonstrated *non-divergence*, never that depth was
*used*. The gap surfaced only later as the depth-is-neutral null on real data (#90) and the
teacher-family instability on the synthetic probe (#99).

**Sharper framing (feeds the depth-null theory).** With a non-identity `F`, a *closed* gate is
locally optimal — opening it raises loss — so the trapped network is *correctly* refusing to use
depth. The problem is not the optimizer; it is the initialization of `F`. See
`docs/concepts/` and the depth-separation write-up.

## 2. Evidence (committed, reproducible)

Two committed artifacts are the empirical backbone of both this spec and the docs:

- **Trap instrumentation** — a deep `MonoResidual` (real layer, `absolute`, `activation="elu"`)
  on a monotone teacher target shows `g_β` pinned at `0.000` and `β` trapped at `≈ −0.0076`
  across training, while `g_α`, block-RMS, and train/test loss confirm the skip path carries the
  fit. (`benchmarks/` diagnostic; to be committed alongside the sweep.)
- **A-vs-B ablation** (`benchmarks/monoresidual_gate_ablation.py`, committed `b19b2a9`) — deep-16
  block stack, monotone teacher target, single deterministic seed. `A` = zero-init `F`'s last
  layer; `B` = `softplus` gate in place of `scaled_elu`:

  | config | g_β | train MSE | test MSE | verdict |
  |---|--:|--:|--:|---|
  | baseline (neither) | **0.000** | 0.865 | 0.908 | trapped |
  | **A** only | 0.194 | 0.405 | 0.423 | escapes trap |
  | **B** only | 0.693 | **≈1e29** | ≈1e29 | **diverges** |
  | **A + B** | 0.751 | 0.130 | 0.134 | best (6.8× over baseline) |

  **Reading:** **A is necessary and primary** (the safety property); **B alone is catastrophic**
  — a dead-zone-free gate starts at `softplus(0)=0.693` and engages a *random* `F` through every
  block → exponential blow-up. This is exactly why the 2026-07-03 spec rejected `softplus`
  ("*softplus(0)≈0.69 — no identity at init*"). **A removes that objection:** with `F` zero-init,
  identity-at-init no longer depends on the gate value, so `softplus` becomes safe *and* supplies
  the clean, non-vanishing gradient the original design wanted. A+B is the coherent resolution,
  not two unrelated patches.

## 3. The fix

### 3.1 (A) Identity-at-init F — the primary, load-bearing change

Zero-initialize the **last layer** of the default residual sub-module `F` (weight and bias). In
every mode this makes `F(x) ≡ 0` at init (`absolute`: `act(x@|0|+0)=act(0)=0` for both convex
`act` and concave `−act(−·)`; `switch`: `act(x@0)−act(x@0)=0`), so the block is an **exact
identity** `y = g_α·skip + g_β·0 = g_α·skip` regardless of the gate value. Intermediate `F`
layers keep their normal init. This is Fixup/ReZero's zero-init-the-branch, adapted: because the
branch output is exactly 0, `∂loss/∂β ≈ 0` at init, so `β` is *not* pushed negative, and the
first useful signal (a perturbation of `F` that lowers loss) pushes `β` **up** — `F` comes online
as §3.1.1 originally intended.

Monotonicity is untouched: zero weights are a valid point of the `|W|`/`W⁺,W⁻` constraint set;
`F` remains non-decreasing.

### 3.2 (B) Dead-zone-free positive gate — beneficial given (A)

Add a `"softplus"` **gate token** (distinct from the existing `softplus` *base activation*) and
make it the default `beta_gate`: `g_β = softplus(β) = ln(1+e^β)`, smooth, `∈ (0,∞)`, gradient
`σ(β) ∈ (0,1)` **everywhere** — no dead zone. `g_β = ln2 ≈ 0.693` at init, which is safe *only*
because (A) makes `F(x)=0` at init. Monotonicity holds (`g_β > 0` for all β).

`α`/`g_α` (skip gate, `shifted_elu`) is **unchanged** — it already satisfies identity-at-init
(`g_α=1`) and has no dead zone on the relevant side; the trap is F-path-only.

### 3.3 Default = A + B (per decision 2026-07-13)

Both A and B become the out-of-the-box `MonoResidual` behaviour in all three backends. A alone is
the minimal *safe* fix; A+B is the best measured result and is what ships. `scaled_elu` is
**retained as a selectable `beta_gate` token** (no removal) for reproducibility of pre-fix runs.

## 4. API & backward compatibility

- **`MonoResidualConfig.beta_gate` default** flips `"scaled_elu"` → `"softplus"`
  (`mononet/core/config.py`); `alpha_gate` unchanged. JSON round-trip carries the token, so old
  configs that pinned `"scaled_elu"` reproduce exactly.
- **`beta_gate` default** flips to `"softplus"` in each `layers.py` and `_kernels.py`
  (torch/jax/keras) and in `reference.py::monotonic_residual`.
- **New gate token `"softplus"`** added to `apply_gate` (reference) and the three kernel gate
  dispatchers. `shifted_elu`/`scaled_elu` remain.
- **Default `F` construction** zero-inits its last layer in all three `MonoResidual`
  implementations. A custom user-supplied `F` is untouched (caller owns it).
- **Compat note.** This changes the numerics of a default `MonoResidual`. Pre-1.0, accepted (same
  posture as the `sub_depth`, `mode="absolute"`, and identity-activation default changes).
  Explicit `beta_gate="scaled_elu"` + no-zero-init escape hatch documented for exact pre-fix
  reproduction.

## 5. Components / repo layout

```
mononet/core/reference.py              # add "softplus" gate token; default beta_gate -> softplus
mononet/core/config.py                 # MonoResidualConfig.beta_gate default -> "softplus"
mononet/{torch,jax,keras}/_kernels.py  # add "softplus" gate branch; default beta_gate -> softplus
mononet/{torch,jax,keras}/layers.py    # default beta_gate -> softplus; zero-init F's last layer
tests/equivalence/cases/*.json         # regenerate: add softplus-gate cases; refresh gated cases
tests/{torch,jax,keras}/test_mono_residual_gate.py   # NEW: gate-opens + monotonicity + zero-init F
benchmarks/monoresidual_gate_ablation.py             # committed A-vs-B reproducer (done, b19b2a9)
benchmarks/monoresidual_gate_trap.py                 # NEW: committed trap instrumentation
docs/concepts/monotonic-residual.md    # AMEND §gate-design with the trap + ablation + rationale
```

## 6. Testing / CI (TDD)

1. **Failing test first — depth is used.** A deep (`sub_depth=2`, ≥12 effective layers)
   `absolute` stack trained a small budget on a target that *requires* depth must (a) open the
   gate (`max g_β` over blocks `> 0.1`, vs baseline `≈ ε`) and (b) beat a matched shallow stack by
   a margin. Fails on `main`, passes after A+B. Per backend (`importorskip`), deterministic seed.
2. **Monotonicity property test still passes** — perturbing any input upward never decreases any
   output, for both size cases and both modes, with softplus gate + zero-init F. The core paper
   guarantee must be preserved.
3. **Zero-init F unit test** — default `F(x) == 0` at init (all modes), so the block is an exact
   identity at init; a custom `F` is not zero-inited.
4. **`softplus` gate token** — reference/kernel parity within tolerance; `g_β > 0`, gradient
   nonzero for `β < 0` (dead-zone-free), `g_β(0) ≈ ln2`.
5. **Equivalence harness** — regenerate committed cases to include `beta_gate="softplus"`; all
   backends agree with the NumPy reference within fixed tolerance.
6. **No-divergence guard** — the A+B deep stack's block-RMS stays `O(1)` at init (guards against
   the B-alone blow-up regressing if zero-init is ever dropped).
7. Full green: `pre-commit --all-files`, strict mypy (`--group bench` too), ruff, docs build.

## 7. Docs (the standing requirement — paper-grade, evidence-backed)

Amend `docs/concepts/monotonic-residual.md` (the existing §3 theory doc). The gate-design
subsection must **explain why the skip-connection gates are chosen the way they are and back it
with data** — not assert it:

1. **Requirements recap** — both gates strictly positive (monotonicity as a hard invariant);
   skip gate `=1` at init; residual branch `≈` off at init for deep stability.
2. **The trap** — why the earlier `scaled_elu`-only design satisfied those requirements yet
   *failed silently*: at-init `∂loss/∂β>0` (random F) + negative-side dead zone ⇒ `g_β` pinned.
   The corrected §3.1.1 reasoning (drift direction was assumed wrong). Include the trap
   instrumentation figure.
3. **The fix and the ablation** — A (identity-at-init F) as the load-bearing safety property; B
   (dead-zone-free softplus) as the enhancement that is *only* safe given A; the A-vs-B table
   (§2) with the B-alone divergence as the evidence that the levers are not independent.
4. **Before/after** — the #90 (large-dataset screen) and #99 (synthetic depth probe) results
   re-run after the fix, showing whether usable depth changes the depth-neutral verdict.
5. **Recommendation** — defaults (A+B), the `scaled_elu` escape hatch, monotonicity preserved.

## 8. Staged plan

- **Stage 1 (this spec):** core + 3 backends + tests + equivalence regen + committed trap/ablation
  reproducers + docs amendment. Success = failing test (1) passes; monotonicity (2) preserved;
  equivalence green.
- **Stage 2:** re-run #99 (probe) and #90 (screen) on the fixed layer; update both draft PRs and
  the docs before/after subsection. Only *then* is the depth-null conclusion trustworthy — the
  prior runs are confounded by the trap.

## 9. Non-goals

- No signed gate / ReZero-style negative-α (monotonicity forbids it).
- No normalization, no scale-control layer (`1/L`, LayerNorm) — the ablation shows the optimizer
  self-regulates `g_β` once A holds; block-RMS is stable. (LayerNorm would also break monotonicity.)
- No removal of `scaled_elu` (kept as a selectable token for reproducibility).
- No change to `MonoLinear`/`MonoInput` or the `switch`/`absolute` math; `g_α`/skip unchanged.
- Not the depth-separation *theory* write-up itself (separate; this fix unblocks its experiments).

## 10. Open items

- Confirm zero-init interacts cleanly with the `absolute` static init (Sub-project A): the last
  layer is overwritten to zero *after* the init runs — verify no init asserts non-zero.
- Decide whether to expose an explicit `zero_init_residual: bool = True` knob or make it implicit
  in the default-`F` path only (leaning implicit: custom `F` opts out by construction).
- Equivalence-case regeneration must stay deterministic (committed JSON, no live seeds).
