# Sub-project A — Unified monotonic layer, three backends, cross-backend equivalence

**Date:** 2026-06-27 (supersedes the original 2026-05-22 Sub-project A draft)
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Papers:**
- Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — arXiv:2205.11775 ([digest](../../references/2205.11775-runje-2023-constrained-mnn.md))
- Sartor, Sinigaglia & Susto, *Advancing Constrained Monotonic Neural Networks*, ICML 2025 — arXiv:2505.02537 ([digest](../../references/2505.02537-sartor-2025-advancing-cmnn.md))

> **Revision note (2026-06-27).** This spec was rewritten from the ground up.
> The original A spec proposed mask-carrying `MonoLinear`/`MonoDense` layers, a
> three-class activation split `(s̆, ŝ, s̃)`, and composed `MonoMLP` /
> `MonoFeatureBlock` models. The design below replaces that with a **two-mode
> unified layer** (`mode ∈ {absolute, switch}`), a **dual-gated monotone
> residual block**, and an **input sign-flip layer** — and drops `MonoMLP`,
> `MonoFeatureBlock`, the three-class split, the `0`/non-monotone mask entry,
> and bounded activations. See §11 for the full list of departures and the
> doc-convention updates this requires.

## 1. Goals & non-goals

### Goals

- Ship a single, framework-idiomatic monotonic dense primitive in two
  constructions selected by `mode`:
  - **`mode="absolute"`** — the ICML 2023 `|W|` weight constraint with a
    two-class convex/concave activation split.
  - **`mode="switch"`** (default) — the ICML 2025 post-activation switch
    `σ(W⁺x+b) − σ(W⁻x+b)`, which needs no activation-split tuning.
- Land it in three idiomatic forms: `mononet.torch.MonoLinear`,
  `mononet.jax.MonoLinear` (Flax NNX), `mononet.keras.MonoDense`, plus the
  NumPy reference `mononet.core.reference.monotonic_dense`.
- Ship `MonoResidual`, a dual-gated monotone residual block that composes a
  monotone sub-module and enables deep monotone residual networks.
- Ship `MonoInput`, a sign-flip layer that maps prescribed per-feature
  monotonicity direction onto the all-non-decreasing primitives.
- Wire up the cross-backend equivalence harness so every backend kernel is
  numerically equal to the NumPy reference (outputs **and** gradients) within
  fixed tolerances.
- Stabilize the public API surface that Sub-projects B/C/D depend on.

### Non-goals

- No bespoke model classes. `MonoMLP` and `MonoFeatureBlock` are **dropped**;
  users compose topologies with native `nn.Sequential` / `nnx.Sequential` /
  `keras.Sequential`.
- No partial-monotonicity handling inside the library. Non-monotone features
  are mixed by a user-supplied external network *before* the first monotone
  layer (§4.4).
- No strict-monotonicity / injective variant for flows (Sub-project D).
- No three-class saturated activation `ρ̃` (§2.2 explains why two classes
  suffice).
- No benchmarks (Sub-project B), no Lean proofs (Sub-project E), no
  training-loop helpers, no ONNX/TFLite export, no CUDA kernels.

## 2. The two constructions in code terms

A layer maps `x ∈ ℝ^{B×n}` to `y ∈ ℝ^{B×m}` with `m = units`, unconstrained
weight `W ∈ ℝ^{n×m}`, bias `b ∈ ℝ^m`, and a base activation `σ` drawn from
the family `Ă` (zero-centred, monotone-increasing, convex, lower-bounded).
**Both constructions are monotonically non-decreasing in every input** — the
layer carries no direction information.

### 2.1 `mode="switch"` (default) — post-activation switch (Sartor Eq. 12)

```
W⁺ = max(W, 0)              # ≥ 0
W⁻ = min(W, 0)              # ≤ 0  (negative part kept negative)
y  = σ(W⁺ᵀx + b) − σ(W⁻ᵀx + b)
```

Both terms are non-decreasing (`σ(W⁺·)` since `W⁺ ≥ 0`; `−σ(W⁻·)` since
`W⁻ ≤ 0`), so `y` is non-decreasing. In the linear regime (`σ = id`) this
reduces to `|W|ᵀx`, not an unconstrained map. The bias is **shared** between
the two terms (dropped when `bias=False`). No activation split is needed:
each neuron already represents a convex and a concave piece.

> Implementation follows Eq. 12, **not** Algorithm 1 of the paper — Algorithm
> 1 is captioned "post-activation" but its printed body is the pre-activation
> form (Eq. 13). See the Sartor digest.

### 2.2 `mode="absolute"` — `|W|` constraint + two-class split (Runje 2023)

```
h = |W|ᵀx + b
y[:, :c] = ρ̆(h[:, :c])          where ρ̆ = σ                 (convex)
y[:, c:] = ρ̂(h[:, c:])          where ρ̂(z) = −σ(−z)         (concave)
c = ceil(convex_fraction · units)
```

`ρ̆` is left-saturating (`∈ S⁻`, e.g. ReLU) and `ρ̂` is its right-saturating
point reflection (`∈ S⁺`). Sartor Prop. 3.9 shows that a convex activation
and its reflection are sufficient for universal monotonic approximation, so
the third **saturated class `ρ̃` of the 2023 paper is intentionally omitted**.
`convex_fraction ∈ [0, 1]`; `0` ⇒ all-concave, `1` ⇒ all-convex.

### 2.3 The dual-gated monotone residual block (`MonoResidual`)

```
y = g_α(α) · skip(x) + g_β(β) · F(x)
```

- `F` is a monotone sub-module (non-decreasing in its inputs). Default: a
  single `MonoLinear`/`MonoDense`. A user-supplied `F` carries the user's
  responsibility for monotonicity.
- `skip(x)` is the identity when `in_features == units`, otherwise a monotone
  linear projection `exp(W_skip)ᵀ · x` with `W_skip ∈ ℝ^{in_features×units}`
  (no activation; positive weights ⇒ non-decreasing) — the analogue of
  torchvision's `downsample` shortcut.
- `g_α, g_β` are **gates** producing non-negative scalars (positivity
  preserves monotonicity). `α, β` are scalar learnable params initialised to
  `0`. With the default gates this is a warm start at ≈ identity
  (`y ≈ skip(x)`) with healthy gradients on both branches:

  | token | function | value@0 | deriv@0 | positive? |
  |---|---|---|---|---|
  | `alpha_gate="shifted_elu"` | `ELU(α)+1` | `1` | `1` | `>0` |
  | `beta_gate="scaled_elu"` | `max(0,β)+ε·exp(min(0,β)/ε)`, `ε=1e-3` | `ε` | `1` | `>0` |

  `max(0,β)` alone would freeze the residual branch (`relu'(0)=0` in
  torch/JAX); the `ε·exp` tail restores a non-zero gradient at `β=0`.

Monotonicity composes: `g_α, g_β > 0`, `skip` and `F` non-decreasing ⇒ block
non-decreasing; a stack of such blocks stays non-decreasing.

## 3. The object-or-literal resolution principle

Every "pick a behavior" argument accepts a **string token** (convenient,
serializable) **or** a native object/callable (powerful, not serializable):

| argument | string token | object / callable |
|---|---|---|
| `activation` | `"relu"`, `"elu"`, … | `ActivationSpec`, or a framework activation callable |
| `init` | `"he_normal"`, … | `InitSpec`, or a framework initializer |
| `alpha_gate` / `beta_gate` | `"shifted_elu"`, `"scaled_elu"`, … | any callable `raw → non-negative tensor` |
| `F` (MonoResidual) | — | a monotone module, or a `factory: units → module` |
| `MonoInput` directions | `+1` / `-1` | `MonotonicityMask` |

Resolution happens inside `__init__`/`build`. **Only string tokens (and
`MonotonicityMask`) round-trip through `MonoConfig`/`MonoResidualConfig`;**
passing a callable opts out of config serialization (standard Keras
custom-object limitation — documented, not worked around).

## 4. Public API surface

```python
from mononet.core.types  import MonotonicityMask, ActivationSpec, InitSpec
from mononet.core.config import MonoConfig, MonoResidualConfig
from mononet.core.reference import monotonic_dense, monotonic_residual

from mononet.torch import MonoLinear, MonoResidual, MonoInput
from mononet.jax   import MonoLinear, MonoResidual, MonoInput
from mononet.keras import MonoDense,  MonoResidual, MonoInput
```

This is the **frozen** surface Sub-projects B/C/D build on.

### 4.1 `MonoLinear` (torch, jax) / `MonoDense` (keras)

```python
MonoLinear(
    units: int,
    mode: Literal["switch", "absolute"] = "switch",
    activation: str | ActivationSpec | Callable = "relu",
    convex_fraction: float = 0.5,        # absolute mode only; ignored in switch
    init: str | InitSpec | Callable | None = None,   # default he_normal
    bias: bool = True,
)
```
- Stores unconstrained `weight` of shape `(in_features, units)` and optional
  `bias` of shape `(units,)`.
- `from_config(cls, cfg: MonoConfig) -> Self` constructor for reproducibility.
- The signature above shows the common tail. **Keras** infers `in_features`
  in `build`, so `MonoDense(units, …)`. **PyTorch** and **Flax NNX** take it
  explicitly as the leading positional, `MonoLinear(in_features, units, …)`,
  matching `nn.Linear` / `nnx.Linear`.

### 4.2 `MonoResidual`

```python
MonoResidual(
    units: int,
    F: Module | Callable[[int], Module] | None = None,   # default: one mono layer
    mode: Literal["switch", "absolute"] = "switch",       # used only for default F
    activation: str | ActivationSpec | Callable = "relu", # used only for default F
    alpha_gate: str | Callable = "shifted_elu",
    beta_gate:  str | Callable = "scaled_elu",
    init: str | InitSpec | Callable | None = None,
)
```
Builds the default `F` from `(units, mode, activation, init)` when `F is None`;
otherwise wraps the supplied module/factory verbatim.

### 4.3 `MonoInput`

```python
MonoInput(directions: int | MonotonicityMask)   # +1, -1, or per-feature mask
```
Negates `−1` columns, passes `+1` columns through. Directions are stored as a
non-trainable buffer / `nnx.Variable` / `get_config` field. This is the only
place prescribed monotonicity *direction* enters the library.

### 4.4 Composing networks (no model classes)

A plain monotone MLP and a monotone ResNet, the native way:

```python
# plain
torch.nn.Sequential(MonoInput(mask), MonoLinear(64), MonoLinear(64), MonoLinear(1, activation="elu"))
# residual
torch.nn.Sequential(MonoInput(mask), MonoResidual(64), MonoResidual(64), MonoLinear(1))
```
Partial monotonicity is handled upstream: a user network mixes non-monotone
features into a monotone representation *before* `MonoInput`.

## 5. `mononet.core`

- **`MonotonicityMask`** — value type restricted to `{-1, +1}` (the `0`
  entry is removed; nothing in the library consumes it). 1-D int8 array.
- **`ActivationSpec`** — `_KNOWN_ACTIVATIONS = {relu, elu, selu, gelu,
  softplus}`. Bounded `tanh`/`sigmoid` are removed (outside `Ă`; the
  reflection/switch math degrades for them). Default `relu`.
- **`InitSpec`** — default scheme `he_normal` (matches ReLU); `glorot_uniform`
  / `lecun_normal` still selectable.
- **`MonoConfig`** (replaces `MonoLinearConfig`) — `{units, mode, activation,
  convex_fraction, init, bias}`, JSON round-trip; gate-free.
- **`MonoResidualConfig`** — `{units, mode, activation, alpha_gate, beta_gate,
  init}`, JSON round-trip; gate fields are string tokens only; a non-default
  `F` is not serialized (documented).
- **`reference.py`** — `monotonic_dense(x, weights, bias, mode, activation,
  convex_fraction)` and `monotonic_residual(...)` are the arithmetic ground
  truth. `monotonic_mlp` is removed.

## 6. Per-backend implementation contract

```
mononet/<backend>/
  _kernels.py    # pure functional ops; native tensors in/out; no state
  layers.py      # MonoLinear/MonoDense, MonoResidual, MonoInput wrappers
```
- `_kernels.py` mirrors the NumPy reference one-to-one (native tensor types).
  The `W⁺/W⁻` (switch) and `|W|` (absolute) reparameterizations and the gate
  functions are computed **inside the forward pass on fresh tensors**; the
  stored unconstrained `weight` parameter is never mutated, so optimizers keep
  updating the free weights.
- **PyTorch:** `nn.Module`, `weight`/`bias` as `nn.Parameter`, gate raws as
  scalar `nn.Parameter`. Optional `torch.compile` smoke test.
- **JAX (Flax NNX):** `nnx.Module`, params as `nnx.Param`. Kernels are
  `jit`/`vmap`/`grad`-clean: the absolute split is slice-and-`concatenate`,
  with no data-dependent branching on tensor values.
- **Keras 3:** `MonoDense(keras.Layer)` using `keras.ops` only; CI default
  `KERAS_BACKEND=jax`, plus one `KERAS_BACKEND=torch` smoke job to confirm the
  ops are backend-agnostic.

## 7. Cross-backend equivalence (`tests/equivalence/`)

### 7.1 Vectors

Committed JSON under `tests/equivalence/cases/{mono_linear,mono_residual,mono_input}/`,
generated once from the NumPy reference (the source of truth) and never
regenerated in CI. Each case stores inputs, `expected_output`,
`expected_grads`, and `tol`. A `tools/regenerate-cases.py` script plus the
reference's git hash accompany the vectors.

### 7.2 Grid (pruned cartesian product, ~50 cases)

| Axis | Values |
|---|---|
| `(batch, in, units)` | `(1,1,1)`, `(4,2,3)`, `(8,7,12)`, `(2,16,1)`, `(3,5,11)` |
| `mode` | `switch`, `absolute` |
| `convex_fraction` (absolute) | `0.0`, `0.5`, `1.0`, and one uneven (`units` not divisible) |
| `activation` | `relu`, `elu`, `selu`, `gelu`, `softplus` |
| `MonoResidual` | `in==units` (identity skip), `in!=units` (projection) |
| `MonoInput` | scalar `+1`/`-1`, mixed mask |
| `dtype` | `float32`, `float64` |

### 7.3 Tests

- `test_*_matches_reference` — backend kernel output vs `expected_output`.
- `test_*_gradients_match` — backend autograd vs reference grads.
- `backend_name` parametrized over `{torch, jax, keras}` via
  `pytest.importorskip`, honoring `MONONET_TEST_BACKEND`.

### 7.4 Property tests (Hypothesis)

Per backend and per mode: for input pairs `x ≤ x′`, assert `y ≤ y′`
componentwise (the non-decreasing guarantee), including through `MonoResidual`
at warm-start and after perturbing the gate raws. A `MonoInput(-1)` →
`MonoLinear` composition is asserted non-increasing in the flipped inputs.

## 8. Default choices

| Choice | Default | Why |
|---|---|---|
| `mode` | `"switch"` | No activation-split tuning; the ICML 2025 result. |
| `activation` | `"relu"` | Sartor's experimental default; unbounded, in `Ă`. |
| `convex_fraction` | `0.5` | Even convex/concave split in absolute mode. |
| `init` | `"he_normal"` | Matches ReLU. |
| `bias` | `True` | Matches `nn.Linear`/`Dense`. |
| `alpha_gate` / `beta_gate` | `"shifted_elu"` / `"scaled_elu"` | Identity warm start with live gradients (§2.3). |

## 9. Backend state & purity notes

The switch decomposition and the gate functions introduce no extra state:
they are pure functions of the stored unconstrained params, recomputed each
forward pass. This keeps `_kernels.py` stateless (the equivalence harness
relies on it) and keeps the JAX path transformation-clean.

## 10. Naming

- PyTorch & JAX mirror their dense layer: `MonoLinear`.
- Keras mirrors its dense layer: `MonoDense`.
- `MonoResidual` and `MonoInput` share one name across backends (no framework
  primitive to mirror).
- NumPy reference uses `snake_case`: `monotonic_dense`, `monotonic_residual`.

## 11. Departures from the original A spec & required doc updates

Removed/changed vs the pre-2026-06-27 design:

- Layers no longer carry a `MonotonicityMask`; direction moves to `MonoInput`.
- `MonotonicityMask` loses its `0` entry (`{-1,+1}` only).
- Three-class split `(s̆, ŝ, s̃)` → two-class `convex_fraction` (absolute);
  `switch` mode needs no split.
- `tanh`/`sigmoid` dropped from `_KNOWN_ACTIVATIONS`.
- `MonoLinearConfig` → `MonoConfig` (+ new `MonoResidualConfig`).
- `monotonic_mlp` reference removed; `monotonic_residual` added.
- `MonoMLP` and `MonoFeatureBlock` dropped entirely.

**Doc-convention updates the implementation plan must make** (not done in this
spec):

- Parent meta-spec [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md):
  update the layout/naming sections that reference `MonoMLP`/`MonoFeatureBlock`
  and `models.py`.
- `CLAUDE.md`: update the "Architecture → Multi-backend pattern" (drop
  `models.py` composed-model bullet), "Naming" (drop `MonoMLP`/
  `MonoFeatureBlock`; add `MonoResidual`/`MonoInput`), and the shared-types
  bullet (`MonoLinearConfig` → `MonoConfig`/`MonoResidualConfig`).

## 12. Open items (decide during planning)

- `"scaled_elu"` token name is mildly misleading for `max(0,·)+ε·exp(…)`;
  keep, or rename (e.g. `"warm_relu"`).
- Gate raws are scalars per block; per-channel gates are a possible future
  extension.
- Whether `MonoResidual` should validate (best-effort) that a user-supplied
  `F` is non-decreasing, or document-only. Lean document-only.
- Whether `MonoInput` needs a `MonoInputConfig` or relies solely on native
  serialization. Lean native-only.

## 13. What is intentionally NOT in this sub-project

- Benchmarks, dataset loaders, training loops (Sub-project B/C).
- Injective / strictly-monotonic primitives and flows (Sub-project D).
- Lean proofs (Sub-project E).
- Partial-monotonicity machinery inside the library (user-side, §4.4).
- Cross-backend save format (rely on each framework's native serialization,
  one round-trip smoke test per backend).
