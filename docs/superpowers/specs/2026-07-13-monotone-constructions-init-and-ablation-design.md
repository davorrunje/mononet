# Monotone constructions (mixed · alternate · split), composition-aware initialization, and the flavor ablation — Design

**Date:** 2026-07-13
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Sub-project:** A (core algorithm / init & naming) + C (the GPU ablation, separable final phase).
**Package area:** `mononet/core/{config,init,reference}.py`, all three backends' `layers.py`
(naming + init application), `tests/equivalence/**` (renamed vectors), `benchmarks/**` (ablation),
`docs/{concepts,benchmarks}/**`.
**Execution note:** the CPU findings below are frozen evidence; the ablation (§8) runs on a **GPU
machine**. This spec is written to be executable end-to-end from there.
**References:**
Runje & Shankaranarayana 2023 ([digest](../../references/2205.11775-runje-2023-constrained-mnn.md));
Sartor et al. 2025 ([digest](../../references/2505.02537-sartor-2025-advancing-cmnn.md)) Prop 3.9;
[absolute-init for deep networks](2026-07-02-absolute-init-deep-networks-design.md);
[deep monotonic residual](2026-07-03-deep-monotonic-residual-design.md);
[Stage-2 unified depth benchmark](2026-07-13-stage2-unified-depth-benchmark-design.md).
Memory: `alternate-construction-init`.

---

## 0. Deliverables & how this spec maps to docs

1. **Core:** a composition-aware initializer in `mononet/core/init.py`, plus the **naming
   migration** `absolute → mixed`, `switch → split`, and a new `alternate` construction.
2. **Research-quality writeup** (§3–§6 below) — the authoritative treatment of the three flavors,
   their initialization, and their residual behavior. **A lighter distillation lands in
   `docs/concepts/`** (constructions page + an initialization page + a residual page). The
   distillation cites this spec; the spec is the source of truth.
3. **Ablation (§8)** — the GPU study; its **results land in `docs/benchmarks/`** as (a) a flavor
   study, (b) an initialization study, (c) a residual study, each across
   activations {relu, elu, softplus, selu}.

Concepts docs = the *why/what* (distilled §3–§6). Benchmarks docs = the *measured* (§8 outputs).

---

## 1. Naming & migration

The `mode` axis is renamed and extended. Old → new:

| old (`mode`) | new (`mode`) | meaning |
|---|---|---|
| `absolute` | **`mixed`** | `\|W\|` weights, within-layer convex/concave split (`convex_fraction`) |
| — (new) | **`alternate`** | `\|W\|` weights, one pure activation class per layer, alternating across depth |
| `switch` | **`split`** | signed decomposition `σ(W⁺x+b) − σ(W⁻x+b)`, no activation split |

`mode: Literal["mixed", "alternate", "split"]`, default `"mixed"` (was `"absolute"`).

**Migration (blast radius is real — audited):**
- `MonoConfig` / `MonoResidualConfig` `Mode` literal + `__post_init__` validation + `to_dict`
  /`from_dict` (which serialize the string).
- All three backends' `layers.py` and `_kernels.py`; `core/reference.py`, `core/init.py`,
  `core/numerics.py`.
- `benchmarks/` flavor plumbing (`search.py::_ALL_FLAVORS`, `flavor_name`, `search_spaces.py`,
  `model_builder.py`) and configs.
- **Committed equivalence vectors** `tests/equivalence/cases/**/*.json` encode `"mode": "switch"`
  *and* the strings appear in **filenames** (`2x16x1-switch-softplus-f32.json`). These must be
  regenerated/renamed, and the `REFERENCE_HASH` guard updated.
- ~40 test files referencing the old strings.
- **Back-compat: none (hard break — user decision 2026-07-14).** The package is `0.0.0a1`
  (alpha); the old strings are removed outright. `from_dict` / constructors on `"absolute"` /
  `"switch"` raise a clear `ValueError` naming the replacement (`"absolute" → use "mixed"`,
  `"switch" → use "split"`). No alias, no `DeprecationWarning`. The equivalence JSON is migrated to
  the new strings so the committed vectors are canonical.

---

## 2. Preliminaries

Notation: input `x ∈ ℝ^d`, a layer maps `ℝ^{n} → ℝ^{m}` with weight matrix `W ∈ ℝ^{n×m}`, bias
`b ∈ ℝ^m`, base activation `ρ̆` from the **convex family `Ă`** (zero-centred, monotone-increasing,
convex, lower-bounded: relu, elu, selu, softplus, …). Define the **concave reflection**
`ρ̂(z) = −ρ̆(−z)` (Runje Eq. 4) — monotone-increasing, concave, upper-bounded, saturating on the
*opposite* side to `ρ̆`. The pair `(ρ̆, ρ̂)` is Sartor's `(σ, σ′)`.

Monotonicity direction (`±1`, unconstrained) is handled once at the input (`MonoInput` sign
flips), so every construction below is stated for the **non-decreasing** case (all inputs `+1`),
exactly as in the current package. `\|W\|` denotes the element-wise absolute-value weight
constraint (`mode` ∈ {mixed, alternate}); `split` uses unconstrained `W`.

---

## 3. The three constructions (research writeup)

### 3.1 Mixed (`\|W\|`, within-layer split) — the 2023 base

A mixed layer applies non-negative weights and splits its `m` output units across the two
activation classes:

```
y = ρ^{cf}(\|W\|ᵀ x + b),   first ⌈cf·m⌉ units use ρ̆ (convex), the rest use ρ̂ (concave)
```

This is Runje & Shankaranarayana's CMFCL (Eq. 7) in `mononet`'s **two-class** variant (the paper's
saturated third class `ρ̃` is dropped; see §3.5). `\|W\| ≥ 0 ⇒ ∂yⱼ/∂xᵢ ≥ 0`, so the layer is
monotone non-decreasing. With `cf = 0.5` (default) each layer carries both a convex and a concave
half; the split is a per-layer hyperparameter, uniform across depth.

### 3.2 Alternate (`\|W\|`, per-layer pure class, alternating across depth) — new

An alternate *stack* fixes one **pure** activation class per layer and alternates it with depth:

```
y_ℓ = ρ_{c(ℓ)}(\|W_ℓ\|ᵀ y_{ℓ−1} + b_ℓ),   c(ℓ) = convex if ℓ even else concave  (or vice-versa)
```

with a linear (`identity`) read-out head (parity-neutral). This is the **literal Sartor Prop 3.9
construction**: non-negative weights, activations alternating `ρ̆ / ρ̂` across depth. It is
expressible today via `mixed` with per-layer `convex_fraction ∈ {1, 0}`, which is precisely why it
is *not* a new weight parametrization — it is an activation **layout** on the same `\|W\|` map. Its
viability hinges entirely on initialization (§4): naively it collapses (§4.5).

### 3.3 Split (signed decomposition) — Sartor 2025

A split layer decomposes the **unconstrained** `W` and applies the activation to each signed part,
sharing the bias (post-activation form, Sartor Eq. 12):

```
W⁺ = max(W, 0) ≥ 0,   W⁻ = min(W, 0) ≤ 0
f̂(x) = σ(W⁺x + b) − σ(W⁻x + b)
```

Both terms are monotone non-decreasing (`W⁺ ≥ 0`; `−σ` of a `W⁻ ≤ 0` map). One layer represents
both a convex and a concave monotone response simultaneously via the `±` split, so there is **no
activation-split hyperparameter** and no `convex_fraction`. In the linear regime it reduces to
`\|W\|x` — not an unconstrained map.

### 3.4 Direction & partial monotonicity

Identical across flavors: prescribed non-increasing inputs are negated at `MonoInput`;
unconstrained (non-monotone) features are handled outside the monotone stack by the user. No
flavor changes the `MonotonicityMask` contract.

### 3.5 Taxonomy — two axes, three flavors

The three are **not** one axis. There are two:

| | activation layout: *mixed* | activation layout: *alternate* |
|---|---|---|
| **weight = `\|W\|`** | **mixed** | **alternate** |
| **weight = signed `W⁺/W⁻`** | **split** (layout N/A — the ± split subsumes it) | — |

`mixed` and `alternate` share the `\|W\|` weight map and differ only in *how the convex/concave
classes are laid out* (within-layer vs across-depth). `split` is a different **weight
parametrization**; it has no activation-layout choice. Presenting all three as peer user-facing
"flavors" is a deliberate UX simplification; the writeup keeps the axes explicit so the
theory is honest.

### 3.6 Universal approximation

- **mixed** — universal for monotone functions. The 2023 proof used the saturated class `ρ̃`;
  `mononet` drops it, justified by Prop 3.9 (alternating convex/reflection is already universal),
  reinterpreted as within-layer mixing (an at-least-as-expressive re-arrangement given width).
- **alternate** — Prop 3.9 *directly*: ≥4 layers of alternating `ρ̆/ρ̂` with non-negative weights
  and a one-side-saturating `ρ̆` is a universal monotone approximator. This flavor is the proof's
  construction verbatim.
- **split** — universal (Sartor Prop 4.1 + stacking ≥4 blocks); strictly more expressive than a
  plain weight constraint (which is its `σ=id` special case).

---

## 4. Initialization (research writeup + findings)

### 4.1 The problem in `\|W\|` networks

Non-negative weights impose a systematic positive drift on the signal: `\|W\| ≥ 0` cannot cancel a
one-sided input, so both the pre-activation mean and variance tend to grow with depth unless the
gain and bias are chosen to counteract it. The existing `absolute_init_params`
([`core/init.py`](../../mononet/core/init.py)) derives, data-free (Gauss-Hermite), a
variance-preserving `gain` and an **output-mean-centering** `bias`, **assuming each layer's
pre-activation is `N(0,1)`**.

### 4.2 Mixed init (current) — works shallow, diverges deep

For `cf = 0.5` the convex/concave contributions cancel, so `bias = 0` and only the gain is solved.
This holds the *single-layer* output at zero mean and unit variance under the `N(0,1)` assumption.
**Empirically it does not survive depth:** at depth 16 a plain mixed stack **diverges at every
learning rate in `{1e-4 … 1e-2}`** for both ReLU and ELU (§6). The `N(0,1)`-per-layer assumption
is not maintained under composition; the per-unit (not per-layer) output means are non-zero and,
under `\|W\| ≥ 0`, accumulate. This is the same deep-`\|W\|`-init difficulty that motivated
[`2026-07-02-absolute-init-deep-networks`](2026-07-02-absolute-init-deep-networks-design.md);
that work is residual-oriented and does not fix deep *plain* mixed stacks. The composition-aware
gain correction (§4.3) does **not** rescue mixed either (§6d) — mixing has no per-layer pure class
to center, so it lacks the alternating-bias stabilizer; the correction is `alternate`-specific.

### 4.3 Alternate init (new: composition-aware, pre-activation-centering)

Abandon output-centering. **Propagate the true signal moments forward and center each
*pre-activation***, giving alternating-sign biases that keep every layer's activation in its live
region. For layer `ℓ` with fan-in `nₗ`, weights `∼ N(0, gₗ²/nₗ)` (so `\|w\|` is half-normal with
`E\|w\| = gₗ√(2/π)/√nₗ`, `E[w²] = gₗ²/nₗ`) and input moments `(mₗ₋₁, vₗ₋₁)`:

```
pre-activation zₗ = Σᵢ \|wᵢ\| hᵢ + bₗ
  E[zₗ]   = gₗ·√(2/π)·√nₗ · mₗ₋₁ + bₗ
  Var[zₗ] = gₗ² · ( vₗ₋₁ + mₗ₋₁²·(1 − 2/π) )
```

1. **Center:** `bₗ = − gₗ·√(2/π)·√nₗ · mₗ₋₁`  ⟹ `E[zₗ] = 0`. Since the convex/concave output mean
   alternates sign with depth, so do the biases — the mechanism that keeps `min(z,0)`/`relu(z)`
   live.
2. **Solve gain:** with `σ_zₗ = gₗ·√(vₗ₋₁ + mₗ₋₁²(1−2/π))`, find `gₗ` s.t.
   `Var[ρₗ(zₗ)] = 1`, `zₗ ∼ N(0, σ_zₗ²)`, via the existing Gauss-Hermite quadrature (convex and
   concave share the gain by symmetry).
3. **Propagate:** `mₗ = E[ρₗ(zₗ)]` (concave: `mₗ = −E[ρ̆(zₗ)]`, the mirror), `vₗ = 1`. Seed with the
   standardized input `(m₀, v₀) = (0, 1)`.

Assumptions (same spirit as the existing init): `\|W\|` independent of incoming activations; `zₗ`
approximately Gaussian by CLT over `nₗ` terms. This is a strict **generalization** of
`absolute_init_params` (its special case: one layer, `N(0,1)` input, output-centering). Sketch:

```python
def alternating_init_params(
    activation: ActivationSpec | str,
    fan_ins: Sequence[int],
    phases: Sequence[bool],          # True = convex layer
) -> list[tuple[float, float]]:      # per-layer (gain, bias)
```

Stays in `core`, pure NumPy, asserted against `core/reference.py`. The recursion needs the
incoming moments, which a bare layer lacks — but the **`prev=` reference (§7)** supplies exactly
the predecessor's stored analytic output mean, so each `alternate` layer resolves its own
`(gain, bias)` at construction from `prev` (and `prev=None` ⇒ the standardized-input entry case).
The batch form `alternating_init_params(activation, fan_ins, phases)` remains as the reference /
test oracle; `prev=` is its per-layer streaming equivalent. **This machinery helps only
`alternate`** — see §6d (mixed/split diverge deep regardless).

### 4.4 Split init

`split` uses **unconstrained `W`** and the standard initializer (`he_normal`/`glorot`/`lecun`);
`absolute_init_params` does not apply. Sartor's argument (§4.2 of the 2025 paper) is that relaxing
`\|W\|` to a signed split makes optimization **less initialization-sensitive**. `mononet`'s CPU
probes only exercised `split` at shallow depth so far (§6); its deep-init behavior is a **primary
question the ablation answers** (§8) — no deep-`split` claim is made here.

### 4.5 The collapse mechanism (why naive alternate dies)

Under the *mixed* init, a naive alternate stack **collapses to a dead gradient** for ReLU and
softplus: a convex layer emits a one-sided (`≥0`) signal; `\|W\| ≥ 0` preserves that sign, so the
next concave layer's pre-activation is `≥ b`; with the mixed/default bias `≈ 0`, the concave ReLU
`min(z,0) = 0` on the whole positive orthant → the layer outputs exactly zero → the stack freezes
at predict-the-mean, identically across all learning rates. ELU survives (its unbounded positive
branch keeps signal alive). The composition-aware init (§4.3) is precisely the fix: centering the
concave layer's pre-activation puts half its mass in the live (`z<0`) region.

---

## 5. Residual connections (research writeup + findings)

### 5.1 `MonoResidual` recap

`y = g_α·skip(x) + g_β·F(x)` with learned gates (`α, β` init 0), a monotone sub-module `F`
(default: `sub_depth=2` monotone layers), and a **near-zero last-layer init** on the default `F`
(`near_zero_scale=1e-3`, bias zeroed) so the block starts near-identity. Exact-zero is avoided
because under `\|W\|` the gradient at `W=0` is `sign(0)=0`, a frozen fixed point.

### 5.2 The skip does not break convexity

A monotone-convex `F` plus a linear skip is still convex, and composing monotone-nondecreasing
convex blocks stays convex (Prop 3.2). So a stack of **all-convex** residual blocks is *still only
convex* — the skip buys nothing on expressivity. Non-convex monotone behavior must come from the
activation layout inside `F` (mixing, alternation, or the split), exactly as in the plain case.

### 5.3 Each flavor in a residual block

- **mixed** — default `F` uses `cf=0.5` layers; near-zero last layer. The shipped path.
- **alternate** — `F = [convex, concave]` (one alternation period per block; skip wraps the
  `ρ̆→ρ̂` pair). A stack of blocks is then a strictly alternating layer sequence with skips every 2
  layers. Because `MonoResidual`'s default `F` hard-wires `cf=0.5` and a custom `F` is **not**
  auto-near-zeroed, the composition-aware init is applied to the custom `F` **and** its final
  (concave) layer is near-zero-scaled to preserve the near-identity start.
- **split** — `F` uses split layers; standard init; near-zero last layer as usual.

### 5.4 Finding — residual is a wash for alternation

The near-identity start tames the depth divergence for **all** flavors, so alternation's plain-deep
advantage largely disappears in the residual setting: at 8 blocks, mixed is slightly better on
ReLU (best 0.021 vs alternate 0.042) and ELU is a tie (~0.04) (§6). **Interpretation:** the
residual near-zero start and the composition-aware init are two *different* cures for the same
depth instability; where residual already applies, alternation adds little. Alternation's unique
value is **deep plain (non-residual) monotone stacks**.

---

## 6. Empirical findings so far (CPU, frozen evidence)

**Setup.** torch CPU; synthetic saturating monotone target `y = Σᵢ wᵢ·sigmoid(3(xᵢ−cᵢ))`
(standardized; predict-the-mean ⇒ MSE ≈ 1.0); `x ∼ U[−1,1]^4`; width 32; Adam; 600–1500 steps;
2–3 seeds. Prototype scripts to be promoted into `benchmarks/`: `/tmp/alt_{smoke,lr,init,deep,fair}.py`.
**Scope caveat:** one target family, one width, torch only, `split` tested only shallow. The
ablation (§8) is the real evaluation; these fix the qualitative phenomena.

**(a) Naive alternate collapses; init-dependent (depth 4, best LR):**

| activation | mixed | alternate (old/mixed init) | alternate (composition-aware init) |
|---|---|---|---|
| relu | 0.231 | 1.02 (dead) | **0.110** |
| softplus | 0.245 | 1.02 (dead) | **0.130** |
| elu | 0.093 | 0.027 | 0.071 |

**(b) Deep plain — the decisive result (depth 16, test MSE; DIV = diverged >1e3):**

| act | kind | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|
| relu | mixed | DIV | DIV | DIV | DIV | DIV |
| relu | alternate | 105.8 | 21.2 | 1.31 | 0.71 | **0.60** |
| elu | mixed | DIV | DIV | DIV | DIV | DIV |
| elu | alternate | 152.8 | 28.5 | **0.45** | 0.45 | 0.45 |

Mixing cannot train a 16-layer plain monotone stack at any tested LR; the composition-aware init
is stable across the whole band.

**(c) Residual — a wash (8 blocks, test MSE):**

| act | kind | 1e-4 | 3e-4 | 1e-3 | 3e-3 | 1e-2 |
|---|---|---|---|---|---|---|
| relu | mixed | 0.096 | 0.046 | 0.027 | **0.021** | 0.084 |
| relu | alternate | 0.863 | 0.508 | 0.341 | 0.143 | **0.042** |
| elu | mixed | 0.147 | 0.059 | **0.042** | 0.043 | 0.083 |
| elu | alternate | 0.191 | 0.096 | 0.081 | 0.074 | **0.040** |

**(d) Composition-aware init helps *only* alternate (depth 16, mean 2 seeds):** applying the
composition-aware gain correction to a *mixed* (`cf=0.5`) stack **still diverges** at every LR
(relu & elu), and *split* also **diverges at depth 16** (standard init; Sartor's reduced
sensitivity does not extend this deep on plain stacks). Only the *alternate* structure is
stabilized — because its stabilizing ingredient is the **alternating-sign bias that centers each
pure layer's pre-activation**, which a within-layer-mixed layer (correct scalar bias ≈ 0) does not
have; gain correction alone is insufficient. **Consequence:** `alternate` + composition-aware init
is the *unique* flavor that trains deep *plain* monotone stacks; `mixed`/`split` need the residual
near-identity start instead. (Caveat: only the natural mixed composition-aware scheme was
tested — gain correction, symmetric bias; a more elaborate per-unit-bias mixed init is unproven,
out of scope.)

**(e) SELU collapse pre-check (init-time diagnostic):** SELU does **not** collapse under naive
alternation — `out.std ≈ 19–23`, healthy gradient norms (~2e4) under both the old and the
composition-aware init, like ELU (the unbounded positive branch keeps signal alive). Only ReLU and
softplus exhibit the hard dead-gradient collapse without the new init. The elevated init-time
variance suggests SELU still benefits from the composition-aware init for depth variance control
(to confirm in §8).

---

## 7. Package / API design

- **Init (core):** add `alternating_init_params` (§4.3) beside `absolute_init_params`. Pure NumPy,
  cross-backend, reference-checked.
- **Construction (`mode`):** rename per §1; add `alternate`. An `alternate` layer's activation
  class (convex/concave) and its composition-aware init are **derived from a predecessor
  reference** (below), *not* from `convex_fraction` — that parameter is **reserved for `mixed`**
  and is rejected in `alternate` mode. No replacement parameter is introduced.

- **The `prev=` design (chosen — user, 2026-07-14).** An `alternate` layer takes an **init-time**
  reference to the preceding monotone layer:

  ```python
  l1 = MonoLinear(d, h, mode="alternate", activation="relu")            # prev=None ⇒ entry (m_in=0), convex
  l2 = MonoLinear(h, h, mode="alternate", activation="relu", prev=l1)   # ⇒ concave (opposite of l1)
  l3 = MonoLinear(h, h, mode="alternate", activation="relu", prev=l2)   # ⇒ convex
  net = nn.Sequential(l1, l2, l3, MonoLinear(h, 1, mode="alternate", activation="identity", prev=l3))
  ```

  At construction the layer reads two things off `prev` — its **phase** (to alternate: opposite of
  `prev`'s) and its **exact analytic output mean** (to center this layer's pre-activation via the
  §4.3 recursion) — computes its own `(gain, bias)` and stores its own analytic output mean for the
  *next* layer, then **discards the reference** (init-time only; nothing stored, no runtime
  coupling, no `state_dict`/pytree impact). Properties:
  - **Auto-entry, no footgun.** `prev=None` *is* the entry declaration (standardized input,
    `m_in=0`). Omitting `prev` cannot silently mis-init — contrast the flag approach, which
    carried a silent ~2× penalty when forgotten (§6, `entry` test) with no safe default.
  - **Auto-alternation.** Phase = opposite of `prev`'s; entry starts convex by default (an optional
    start-phase override lives on the entry layer only — pin in the plan).
  - **Exact, not fixed-point.** The `prev` chain propagates true moments from the entry down (this
    is §4.3, plumbed through references instead of a builder loop).
  - **Cross-backend.** It is an object reference consumed at `__init__`, needing no runtime
    introspection. *Runtime* peeking at a predecessor is **not** portable — torch/NNX modules have
    no sibling/parent back-reference; Keras exposes only the private functional-API
    `_keras_history` — so the explicit init-time reference is the portable mechanism. **Keras
    nuance:** the mean chain is fan-in-independent (resolved at construction), while the fan-in-
    scaled bias is applied in `build()` when the input shape is known.
  - **Residual.** In a `MonoResidual` block `F = [convex, concave]`, the second layer takes
    `prev=` the first, and its final (concave) layer is additionally near-zero-scaled to preserve
    the near-identity start (a custom `F` is not auto-near-zeroed).

- **Prototype-validated (torch, 2026-07-14).** A subclass of the real `MonoLinear` confirmed:
  auto-alternating phases (`CcCcCc…`), entry bias exactly `0` (`prev=None`), alternating-sign
  interior biases (±4.94), `prev` neither retained nor in `state_dict`, monotonicity preserved
  (finite-difference), and reproduction of the stable depth-16 numbers (relu 0.58, elu 0.45).
  Script: `/tmp/prev_proto.py`.

- **`prev=` is `alternate`-only.** Composition-aware init benefits neither `mixed` nor `split`
  (§6d: both diverge at depth 16 regardless); `mixed`/`split` self-initialize per-layer exactly as
  today. Passing `prev=` with `mode ∈ {mixed, split}` raises `ValueError`.

- **Forward kernels & the monotonicity guarantee are unchanged.** The init only sets initial
  weight/bias values; `\|W\|` (mixed/alternate) and the `±` split (split) are untouched, so the
  equivalence harness continues to validate the math.

---

## 8. Ablation study (GPU, separable phase)

Built on the Stage-2 infrastructure and result schema. **Full factorial** as requested:

| factor | levels |
|---|---|
| **flavor** | mixed · alternate · split |
| **activation** | relu · elu · softplus · **selu** |
| **topology** | plain · residual |
| **depth** | 4 · 8 · 16 (plain layers / residual blocks) |
| **init** | flavor-appropriate, **and for `alternate` sweep BOTH inits** (composition-aware *and* the existing output-centering `absolute_init_params`) × every activation, so the per-activation init choice — the ELU question of §12 — is decided from data. mixed→`absolute_init_params`; split→standard |

**Pre-check (gating):** a fast collapse/monotonicity screen for **every (flavor, activation)** at
init — assert non-zero output variance and non-vanishing gradient (catch any SELU/softplus
collapse) and monotonicity (finite-difference). Any collapsing cell is reported and excluded with
an explicit `log`, never silently dropped.

**Metrics.** Stage-2 primary (ROC-AUC classification / MSE regression) + **seed dispersion** (IQR)
+ **convergence** (epochs-to-best) + an explicit **stability readout**: the fraction of (seed, LR)
runs that diverge — the deep-plain robustness result is the headline, so divergence is a
first-class measured quantity, not a footnote.

**Datasets.** The Stage-2 set (10 real + synthetic ladder); LR sensitivity reported on 2
representative datasets (one classification, one regression).

**Benchmark docs produced (→ `docs/benchmarks/`):**
1. **Flavor study** — mixed vs alternate vs split across activations/topology/depth: primary
   metric + dispersion + convergence.
2. **Initialization study** — per flavor, variance-preservation and **divergence-rate vs depth**;
   the deep-plain mixed-divergence vs alternate-stability table; split's deep-init behavior
   (the open question of §4.4), across all four activations.
3. **Residual study** — each flavor in `MonoResidual` vs its plain counterpart at matched
   depth/params, quantifying §5.4 across all four activations.

---

## 9. Docs plan (→ `docs/concepts/`)

Lighter distillations of the writeup, cross-linking to this spec and the benchmark results:
- **Constructions** page — §3 distilled (the three flavors, the two-axis taxonomy, UAP one-liners).
- **Initialization** page — §4 distilled for all three flavors, including the collapse mechanism
  and the composition-aware fix, with the deep-plain divergence result summarized.
- **Residual** page — §5 distilled for all three flavors; fill the "Depth on real data" placeholder
  left by the MonoResidual work with the §5.4 finding once the ablation confirms it on real data.

---

## 10. Testing

- **Init math (core, CI):** `alternating_init_params` reproduces the moment recursion; matches a
  Monte-Carlo estimate of per-layer output variance (≈1) and mean within tolerance; reduces to
  `absolute_init_params` in the single-layer standardized case.
- **Collapse regression:** alternate ReLU/softplus stacks built with the new init have non-zero
  output variance and non-vanishing gradient at init (assert the contrast vs the mixed/default init
  which collapses).
- **Variance preservation at depth:** per-layer output variance ≈1 through depth 16.
- **Monotonicity:** finite-difference non-decreasing check for every (flavor, activation) stack.
- **Naming migration:** `from_dict` / constructors **raise `ValueError`** on old `absolute`/
  `switch` (message names the replacement); new strings round-trip; **equivalence vectors
  regenerated** and the `REFERENCE_HASH` guard updated; all backends agree (equivalence harness)
  under the new names.
- **Determinism / cross-backend:** same seed → same params; the NumPy init drives torch/jax/keras
  identically.
- Full green: `pre-commit`, strict mypy, ruff, docs build (`-W`).

---

## 11. Non-goals

- **No new weight parametrization.** `alternate` is a layout on the existing `\|W\|` map; `split`
  is the renamed `switch`. Only one new `mode` *value*, no new kernel.
- **No residual-alternation feature.** Evidence says residual is a wash for alternation (§5.4); we
  measure it in the ablation but do not add a dedicated residual-alternate construction.
- **No default change** beyond the rename: default `mode` is `mixed` (formerly `absolute`),
  default init path unchanged for mixed/split; composition-aware init is opt-in via `alternate`.
- **No softmax activation** (not a pointwise monotone activation; out of the family `Ă`).
- **No new metric** beyond Stage-2's + the divergence-rate readout.

---

## 12. Open items (for user review)

- **§7 application surface — *resolved* (user, 2026-07-14):** the `prev=` init-time reference
  (alternate-only; auto-entry, auto-phase, exact moment propagation, cross-backend). No builder,
  no composed class, `convex_fraction` untouched.
- **ELU init path — decided by the ablation (user, 2026-07-14).** ELU is well-served by the
  *existing* init and slightly better there than the composition-aware one (§6a). Do **not** decide
  now; §8 sweeps both inits × all activations for `alternate`, and the per-activation init default
  is chosen from that data.
- **SELU viability** under alternation — *resolved* (§6d): no collapse; included in the sweep.
- **Reconcile with `2026-07-02-absolute-init-deep-networks`** so there is one coherent deep-init
  story (mixed deep-init vs the composition-aware init), not two.
- **Phase convention** (start convex vs concave) and head parity — pin in the plan.
- **Deprecation window** — *resolved:* none; hard break (§1).

---

## 13. Sequencing

1. **Naming migration (core + backends + tests + equivalence vectors)** — one PR, CI-green
   (hard break + regenerated vectors + `REFERENCE_HASH`). No behavior change.
2. **Composition-aware init (`alternating_init_params`) + tests (§10)** — one PR.
3. **Application surface (§7) + concepts docs (§9)** — same or follow-up PR, per chosen option.
4. **Ablation (§8)** — separate spec → plan → GPU run → benchmark docs. Gated on 1–3 landing.
