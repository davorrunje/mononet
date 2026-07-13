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
> monotone and non-divergent at depth, but the residual branch `F` never engages, so the "deep"
> stack is effectively shallow — the root cause of the depth-null results (#90, #99). The ablation
> shows this is **two independent traps**, needing two fixes: **(A)** *near-zero* init of `F`'s
> last layer (scale ≈ `1e-3`, not exact-zero) so `F ≈ 0` at init yet its weights stay trainable,
> and **(B)** a dead-zone-free positive gate (`softplus`) so `g_β` can open. Both ship as the new
> default for `MonoResidual` across all three backends, with **paper-grade docs justifying the
> gate/skip design, backed by the trap instrumentation + the A-vs-B ablation + before/after
> re-runs**.

## 1. Problem: two independent traps starve the residual branch

`MonoResidual` computes `y = g_α(α)·skip(x) + g_β(β)·F(x)` with strictly-positive gates
(monotonicity requires `g_α, g_β ≥ 0`; a signed gate would flip a branch to non-increasing).
The default F-path gate token is `scaled_elu`: `g_β = max(β,0) + ε·exp(min(β,0)/ε)`, `ε=1e-3`.
The ablation (§2) shows the residual branch `F` fails to engage for **two independent reasons**:

**Trap 1 — gate dead zone.** The approved 2026-07-03 spec (§3.1.1) claims the `ε·exp(β/ε)` tail
gives "*a small but nonzero gradient … so β can escape 0 and F can come online.*" **It does not.**
The claim assumes `β` drifts *up* (engaging `F` helps). At init `F` is random, so engaging it
*raises* loss ⇒ `∂loss/∂β > 0` ⇒ descent drives `β` **down**; on the negative side `scaled_elu`'s
gradient `exp(β/ε)` collapses (at the observed trapped `β ≈ −0.0076`, gradient `≈ e^{−7.6}`), so
`g_β` is pinned at `≈ε` **forever**. Confirmed: with `scaled_elu`, `g_β = 0.000` at any F-init.

**Trap 2 — `|W|` frozen-weight fixed point.** Suppose we sidestep Trap 1 the naive (Fixup) way by
**exact-zero**-initializing `F`'s last layer. Under the `absolute` construction `F` uses `|W|`,
and `d|W|/dW` at `W=0` is `sign(0)=0` — a **gradient fixed point**. So the zeroed weights **never
move**: `F` degenerates to a per-block learned *constant* (only the bias moves), not an
`x`-dependent depth function. Confirmed: exact-zero → F's last-layer weights move in `0/16` blocks;
train MSE floors at `0.14` (constants only) vs `0.07` when `F` genuinely trains.

**Why it stayed invisible.** The 2026-07-03 §2 sweep reported depth-32 stacks training to
MSE ≈ 0.08–0.10 and concluded "residual skips fully fix deep trainability." Correct, but answering
the wrong question: on a shallow-learnable target the stack trains **via the skip path** with `F`
off. It showed *non-divergence*, never depth *use*. The gap surfaced later as the depth-neutral
null on real data (#90) and the teacher-family instability on the probe (#99). Feeds the
depth-null theory: a closed gate / constant `F` is locally optimal, so the net *correctly* refuses
depth — the defect is in `F`'s init and gate, not the optimizer.

## 2. Evidence (committed, reproducible)

Two committed artifacts are the empirical backbone of both this spec and the docs:

- **Trap instrumentation** (`benchmarks/monoresidual_gate_trap.py`, to be added) — a deep
  `MonoResidual` (real layer, `absolute`, `activation="elu"`) on a monotone teacher target shows
  `g_β` pinned at `≈0`, `β` trapped negative, and `g_α`/block-RMS/loss confirming the skip path
  carries the fit while `F` sits idle.
- **A-vs-B ablation** (`benchmarks/monoresidual_gate_ablation.py`, committed) — depth-16 stack,
  monotone teacher, deterministic seed. `A` = F-last-layer init `∈ {off, exactzero,
  nearzero(×1e-3)}`; `B` = gate `∈ {scaled_elu, softplus}`. `F-moved` = # of 16 blocks whose
  last-layer weights left their init:

  | A (F init) | B (gate) | g_β | train MSE | F-moved | verdict |
  |---|---|--:|--:|:--:|---|
  | off | scaled_elu | 0.000 | 0.865 | — | gate dead-zone trap |
  | exactzero | scaled_elu | 0.194 | 0.405 | 0/16 | F frozen → constant |
  | nearzero | scaled_elu | 0.000 | 0.994 | 16/16 | still gate-trapped |
  | off | softplus | 0.693 | **≈1e28** | — | **diverges** (random F engaged) |
  | exactzero | softplus | 0.753 | 0.136 | 0/16 | gate opens, F still frozen |
  | **nearzero** | **softplus** | **0.700** | **0.068** | **16/16** | **best (A+B, the fix)** |

  **Reading.** The two traps are independent and need independent fixes. **`softplus` (B) opens
  the gate** — with `scaled_elu` the gate stays shut regardless of init (rows 1, 3). **Near-zero
  init (A) lets `F` learn `x`-dependence** — exact-zero freezes the weights (rows 2, 5: `F-moved
  0/16`, MSE floored at 0.14), near-zero trains them (row 6: `16/16`, MSE 0.068). Neither lever
  alone works: `nearzero+scaled_elu` is trapped (0.994), `off+softplus` diverges. Only
  `nearzero+softplus` both opens the gate **and** trains `F`. This retires the 2026-07-03
  rejection of `softplus` ("*softplus(0)≈0.69 — no identity at init*"): near-zero `F` makes
  identity-at-init independent of the gate value, so `softplus` is safe.

## 3. The fix

### 3.1 (A) Near-zero init of F — frees the weights from the `|W|` fixed point

Initialize the **last layer** of the default residual sub-module `F` by **scaling its normal-init
weight by a small factor `_NEAR_ZERO_SCALE = 1e-3`** and zeroing its bias. This keeps `F(x) ≈ 0`
at init (init F-output RMS ≈ `0.03`; the block starts ≈ identity `y ≈ g_α·skip`, deep-stack-safe)
while keeping the weights **nonzero**, so `sign(W) ≠ 0` and gradients flow — `F` learns
`x`-dependence. Intermediate `F` layers keep normal init. This is Fixup's zero-init-the-branch
**adapted to the `|W|` constraint**: exact-zero is a gradient fixed point here (Trap 2), so we use
*near*-zero. The scale has a stable band — `1e-3` gives F-RMS ≈ 0.03 and trains; `≥1e-2` lets `F`
engage too strongly at init and the deep stack blows up (same failure as `off+softplus`). `1e-3`
is the chosen default.

Monotonicity is untouched: scaled weights are a valid point of the `|W|`/`W⁺,W⁻` constraint set;
`F` remains non-decreasing.

### 3.2 (B) Dead-zone-free positive gate — opens the gate

Add a `"softplus"` **gate token** (distinct from the existing `softplus` *base activation*) and
make it the default `beta_gate`: `g_β = softplus(β) = ln(1+e^β)`, smooth, `∈ (0,∞)`, gradient
`σ(β) ∈ (0,1)` **everywhere** — no dead zone, so `β` can move off init in either direction and the
gate opens. `g_β = ln2 ≈ 0.693` at init, safe *because* (A) keeps `F(x) ≈ 0` there. Monotonicity
holds (`g_β > 0` for all β).

`α`/`g_α` (skip gate, `shifted_elu`) is **unchanged** — it satisfies identity-at-init (`g_α=1`),
has no dead zone on the relevant side, and its skip path is already `x`-dependent; both traps are
F-path-only.

### 3.3 Default = A + B (per decision 2026-07-13)

Both A and B become the out-of-the-box `MonoResidual` behaviour in all three backends. A alone is
the minimal *safe* fix; A+B is the best measured result and is what ships. `scaled_elu` is
**retained as a selectable `beta_gate` token** (no removal) for reproducibility of pre-fix runs.

## 4. API changes

The package has **never been publicly released**, so there is no backward-compatibility
constraint — we change the defaults outright, no deprecation window or compat shims.

- **`beta_gate` default** flips `"scaled_elu"` → `"softplus"` everywhere it is declared:
  `mononet/core/config.py` (`MonoResidualConfig`), each backend `layers.py` and `_kernels.py`
  (torch/jax/keras), and `reference.py::monotonic_residual`. `alpha_gate` (skip gate) is
  unchanged.
- **New gate token `"softplus"`** added to `apply_gate` (reference) and the three kernel gate
  dispatchers. `shifted_elu` and `scaled_elu` remain selectable tokens — `scaled_elu` is kept
  only so the trap/ablation experiments can still instantiate the old gate for comparison, not
  for user compatibility.
- **Default `F` construction** *near-zero*-inits its last layer in all three `MonoResidual`
  implementations — scale the normal-init weight by `_NEAR_ZERO_SCALE = 1e-3` and zero the bias,
  so `F(x) ≈ 0` at init but the weights stay trainable (not exact-zero; see §3.1 / Trap 2). A
  custom user-supplied `F` is untouched (the caller owns its init).

## 5. Components / repo layout

```
mononet/core/reference.py              # add "softplus" gate token; default beta_gate -> softplus
mononet/core/config.py                 # MonoResidualConfig.beta_gate default -> "softplus"
mononet/{torch,jax,keras}/_kernels.py  # add "softplus" gate branch; default beta_gate -> softplus
mononet/{torch,jax,keras}/layers.py    # default beta_gate -> softplus; near-zero-init F's last layer
tests/equivalence/cases/*.json         # regenerate: add softplus-gate cases; refresh gated cases
tests/{torch,jax,keras}/test_mono_residual_gate.py   # NEW: gate-opens + monotonicity + near-zero F
benchmarks/monoresidual_gate_ablation.py             # committed A-vs-B reproducer (done)
benchmarks/monoresidual_gate_trap.py                 # NEW: committed trap instrumentation (writes JSON)
benchmarks/results/monoresidual-gate/*.json          # NEW: committed trap + ablation results
docs/concepts/monotonic-residual.md    # REWRITE gate/skip requirements + design + experiments (see §7)
```

Both experiments follow the `benchmarks/deep_residual_run.py` pattern (a `python -m` runnable
module that writes a committed results JSON and prints a table); the docs render from the
committed JSON with a one-line reproduce command, so every number on the page is reproducible.
`monoresidual_gate_ablation.py` (already committed) will be adjusted to also emit its JSON.

## 6. Testing / CI (TDD)

1. **Depth is used (the headline regression).** A deep (`sub_depth=2`, ≥12 effective layers)
   `absolute` stack trained a small budget on a target that *requires* depth must open the gate
   (`max g_β` over blocks `> 0.1`, vs the trap's `≈ ε`) and train below a fixed MSE floor.
   Decisive guard: `g_β > 0.1` is impossible under either trap (dead-zone pins it at `ε`; the
   result was verified red on `main`). Torch, deterministic seed.
2. **F's weights actually train (Trap-2 guard).** After a few steps on the deep default stack,
   the last-layer weight of every `MonoResidual`'s `F` has moved from its init (`|Δ| > 0`) — pins
   near-zero init against a regression to exact-zero (which would freeze them). Per backend.
3. **Near-zero init unit test** — default `F`'s last-layer weight is small but **nonzero** at init
   (`0 < ‖W_last‖ ≪ ‖W_last‖_normal`), its bias is zero, and `F(x)` RMS `≈ 0` (block ≈ identity);
   a custom `F` is untouched. Per backend.
4. **Monotonicity property test still passes** — perturbing any input upward never decreases any
   output, both size cases and both modes, with softplus gate + near-zero F. The core paper
   guarantee must be preserved.
5. **`softplus` gate token** — reference/kernel parity within tolerance; `g_β > 0`, gradient
   nonzero for `β < 0` (dead-zone-free), `g_β(0) ≈ ln2`.
6. **Equivalence harness** — regenerate committed cases to include `beta_gate="softplus"`; all
   backends agree with the NumPy reference within fixed tolerance.
7. **No-divergence guard** — the A+B deep stack's block-RMS stays `O(1)` at init (guards against
   the scale-too-large / `off+softplus` blow-up).
8. Full green: `pre-commit --all-files`, strict mypy (`--group bench` too), ruff, docs build.

## 7. Docs (the standing requirement — paper-grade, evidence-backed)

Target page: **`docs/concepts/monotonic-residual.md`** — "Deep monotonic networks with residual
skips" (already in the Concepts toctree). It currently contains a **refuted** rationale (the
"Why the gates are shaped this way" subsection, lines ~52–60: "*the `ε·exp(β/ε)` tail … so `β`
can escape `0` and `F` can come online*") and an experiments section whose "trains to MSE ≈ 0.1"
is read as "depth works" when it only shows non-divergence. Both must be corrected. The page must
**document the requirements, the design choices, the experiments, and why we chose A+B — backed
by reproducible benchmarks**, not assertion. Required structure:

1. **Requirements for skip connections and gates** (make the constraints explicit and complete):
   - *Skip path* — must be **monotone** (identity when `in==out`; `exp`-parametrized positive
     projection when `in≠out`) and **near-identity at init** (strongest warm start; keeps deep
     stacks forward-stable). Its gate `g_α` must be strictly positive and `=1` at init.
   - *Residual path `F`* — must be **monotone** (holds by the `|W|`/`switch` construction for
     *any* weights); **contribute ≈ 0 at init** so the block starts as an identity and the deep
     stack does not blow up; and its **weights must stay trainable** at init (this is the subtle
     one — exact-zero fails it under `|W|`). Its gate `g_β` must be strictly positive
     (monotonicity) — which **rules out a signed/ReZero-style gate** — and must be able to
     **open** (leave its init value) once `F` becomes useful.
   - *Why positivity is non-negotiable* — a negative gate flips a branch to non-increasing;
     monotonicity is enforced at call time via the gate parametrization, so it is a hard
     invariant under free optimization (keep the existing theorem + proof).
2. **Design choices** — the two independent traps and the one knob each fixes:
   - Skip gate `g_α = elu(α)+1` (unchanged): `=1` at init, unbounded, decays to `0⁺`; why
     `sigmoid`/`exp`/`softplus` fail *for the skip gate*.
   - **Trap 1 (gate dead zone) → residual gate `g_β = softplus(β)`:** strictly positive, **no dead
     zone** (gradient `σ(β)∈(0,1)` everywhere), so `β` moves off init and the gate opens. The old
     `scaled_elu` pinned `g_β` at `ε` because its negative-side gradient vanishes and a random `F`
     pushes `β` negative.
   - **Trap 2 (`|W|` frozen-weight fixed point) → near-zero init of `F`'s last layer:** "contribute
     ≈ 0 at init" is achieved **by near-zero initialization (scale `1e-3`), not exact-zero and not
     by shrinking the gate.** Exact-zero is a `sign(0)=0` gradient fixed point under `|W|`, so the
     weights freeze and `F` becomes a constant; near-zero keeps them trainable. Decoupling
     identity-at-init from `g_β` is what lets us use the clean gate — explain that `softplus` was
     previously rejected *because* `softplus(0)≈0.69` broke identity-at-init, and near-zero `F`
     removes exactly that objection. Note the stable scale band (`≈1e-3`; `≥1e-2` re-blows-up).
3. **Experiments (reproducible under `benchmarks/`)** — replace the "depth works because it
   trains" framing:
   - Keep the skip-K sweep (it correctly establishes skips fix **divergence**), but reframe it as
     *forward stability*, not depth-utilisation.
   - **Trap instrumentation** (`benchmarks/monoresidual_gate_trap.py`): `g_β` pinned at `≈0`, `β`
     trapped negative, skip path carrying the fit — evidence that depth went *unused*.
   - **A-vs-B ablation** (`benchmarks/monoresidual_gate_ablation.py`): the six-row table (§2) —
     the two independent traps, exact-zero freezing `F` (`F-moved 0/16`), `off+softplus`
     divergence, and `nearzero+softplus` as the only config that both opens the gate and trains
     `F`.
   - **Before/after** (#90 large-dataset screen, #99 synthetic depth probe) re-run on the fixed
     layer — whether usable depth changes the depth-neutral verdict.
   Each renders from committed `benchmarks/results/**` JSON with a one-line `uv run … -m …`
   reproduce command.
4. **Why A+B** — a short synthesis: two independent traps ⇒ two independent fixes; `softplus`
   opens the gate, near-zero init frees `F` to learn; neither alone suffices (near-zero+scaled_elu
   is trapped, off+softplus diverges); monotonicity preserved throughout.
5. **Recommendation** — the shipped defaults (A+B), `sub_depth=2`, monotonicity guarantee intact.

## 8. Staged plan

- **Stage 1 (this spec):** core + 3 backends + tests + equivalence regen + committed trap/ablation
  reproducers + docs amendment. Success = failing test (1) passes; monotonicity (2) preserved;
  equivalence green.
- **Stage 2 (the benchmark re-run):** the fix changes whether depth is usable, so every
  depth-sensitive result is confounded by the trap and must be re-run on the fixed layer.
  Scope and execution:
  - **All 10 datasets in the registry** — the paper 5 (`auto`, `heart`, `compas`, `loan`,
    `blog`) **and PR #90's additions** (`adult`, `taiwan`, `polish`, `german`, `lc` =
    Lending Club). No dataset is exempt; the depth-neutral verdict must be re-tested everywhere.
  - **Both GPUs (5090 + 3090)** via the existing device-assigning launcher pattern
    (`benchmarks/screen_launch.py` / `loan_ladder_launch.py`), `n_jobs=1` per process (the
    threaded-Optuna deadlock stands — parallelize across *processes/GPUs*, not threads).
  - **Size-driven batch bands, for speed.** Generalize `benchmarks/_common/search_spaces.py`'s
    hardcoded `_LARGE_BATCH_DATASETS = {"loan", "blog"}` into a **rule keyed on train-set size**:
    datasets above a row threshold draw from the large-batch band (`512–4096`) so 50-epoch
    training is launch-bound-cheap; small datasets (`heart`, `auto`, `german`, …) keep the
    small band (`8–256`) where small batches still matter for generalization. The threshold is
    derived once from the loaded `n_train` (not a hand-maintained name set), so new datasets are
    banded automatically. `lc`/`adult`/`taiwan` land in the large band; `heart`/`auto`/`german`
    in the small band.
  - **Artifacts:** the standard flavor-comparison / deep-residual-accuracy table and the #90
    screen, both regenerated across all 10 datasets; plus the #99 synthetic probe. Update both
    draft PRs and the docs before/after subsection. Only *then* is the depth conclusion
    trustworthy.

## 9. Non-goals

- No signed gate / ReZero-style negative-α (monotonicity forbids it).
- No normalization, no scale-control layer (`1/L`, LayerNorm) — the ablation shows the optimizer
  self-regulates `g_β` once A holds; block-RMS is stable. (LayerNorm would also break monotonicity.)
- No removal of `scaled_elu` (kept as a selectable token so the trap/ablation experiments can
  still instantiate the old gate — not for user compatibility).
- No change to `MonoLinear`/`MonoInput` **math** or the `switch`/`absolute` kernels; `g_α`/skip
  unchanged. (Near-zero init is a post-construction weight rescale in `MonoResidual`, not a kernel
  change.)
- Not the depth-separation *theory* write-up itself (separate; this fix unblocks its experiments).

## 10. Open items

- Near-zero init is a rescale applied *after* the `absolute` static init runs (Sub-project A) —
  verify no init path asserts a specific weight norm afterwards.
- `switch` mode: verify near-zero init also frees the weights there (`W⁺=max(W,0)`, `W⁻=min(W,0)`
  have the same `sign(0)=0` fixed point) — the ablation covers `absolute`; the property/regression
  tests must cover `switch` too.
- Near-zero init is **implicit in the default-`F` path only** (a custom `F` opts out by
  construction); no separate `zero_init_residual` knob — a custom `F` already owns its init.
- `_NEAR_ZERO_SCALE = 1e-3` is the shared constant across backends; keep it in one place per
  backend and document the stable band in the tests' comments.
- Equivalence-case regeneration must stay deterministic (committed JSON, no live seeds).
