# Sub-project A — Core algorithm, three backends, cross-backend equivalence

**Date:** 2026-05-22
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Paper:** Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>

## 1. Goals & non-goals

### Goals

- Implement the Constrained Monotone Fully Connected Layer (CMFCL, eq. 7 in the paper) as the canonical primitive of this library.
- Land it in three idiomatic forms — `mononet.torch.MonoLinear`, `mononet.jax.MonoLinear` (Flax NNX), `mononet.keras.MonoDense` — plus the framework-agnostic `mononet.core.reference.monotonic_dense` already stubbed in the repo.
- Land both architectures from the paper: **Type 1** (`MonoMLP`, Fig. 4) and **Type 2** (`MonoFeatureBlock`, Fig. 5), per backend.
- Wire up the cross-backend equivalence harness (`tests/equivalence/`) so any backend kernel is asserted numerically equivalent to the NumPy reference within fixed tolerances; gradients too.
- Stabilize the public API surface that Sub-projects B/C/D will all depend on. After this spec lands, importing `MonoLinear`, `MonoDense`, `MonoMLP`, `MonoFeatureBlock` and getting working layers is the v0.1.0 bar.

### Non-goals

- No benchmarks against the paper's datasets (that's Sub-project B).
- No invertible / strictly-monotonic variant for flows (that's Sub-project D — uses a different, stricter parameterization).
- No formal proofs (that's Sub-project E).
- No training-loop helpers; users compose layers into their own training loops.
- No CUDA-specific kernels — backends use stock framework ops; CUDA acceleration comes for free from each framework.
- No ONNX / TFLite export.
- No automatic determination of the `s` activation split — user-provided, with documented defaults.

## 2. The algorithm in code terms

For a Constrained Monotone Fully Connected Layer with `in_features = n`, `out_features = m`:

```
y = ρ_s( |Wᵀ|_t · x + b )

where
  W ∈ ℝ^{n×m},  b ∈ ℝ^m,  x ∈ ℝ^n,  y ∈ ℝ^m
  t ∈ {-1, 0, +1}^n            # monotonicity mask, per input feature
  s = (s̆, ŝ, s̃) ∈ ℕ³, s̆ + ŝ + s̃ = m   # activation split
  ρ̆ ∈ Ă : zero-centred, monotone-increasing, convex, lower-bounded
                              # (e.g. ReLU, ELU, SELU, GELU)
  ρ̂(x)  = -ρ̆(-x)              # concave reflection
  ρ̃(x)  = ρ̆(x+1) - ρ̆(1)        if x < 0
        = ρ̂(x-1) + ρ̆(1)        if x ≥ 0      # saturated piecewise
  ρ_s(h)_j = ρ̆(h_j)                          if j ≤ s̆
           = ρ̂(h_j)                          if s̆ < j ≤ s̆ + ŝ
           = ρ̃(h_j)                          otherwise
  ( |M|_t )_{j,i} =   |m_{j,i}|              if t_i = +1
                    = -|m_{j,i}|             if t_i = -1
                    =  m_{j,i}               if t_i =  0
```

In code, the layer takes three things the user must choose:

1. `monotonicity: MonotonicityMask` (length `n`) — already in `mononet.core.types`.
2. `activation: ActivationSpec` — the base ρ̆ (name only; backend resolves).
3. `activation_split: ActivationSplit` — the new piece, `(s̆, ŝ, s̃)` such that the three sum to `out_features`.

The `MonotonicityMask` and `ActivationSpec` types already exist. **`ActivationSplit` is new** and is the most important addition for this sub-project.

## 3. Additions to `mononet.core`

### 3.1 New: `ActivationSplit` in `mononet.core.types`

```python
@dataclass(frozen=True, slots=True)
class ActivationSplit:
    """Per-layer split into (convex, concave, saturated) neuron counts.

    Sums must equal the layer's `out_features`. Sentinel value `None`
    means "use the default split for this `out_features`" — equal thirds
    rounded to favour the convex bucket when `out_features` is not
    divisible by 3.
    """
    convex:    int   # s̆ — neurons that get ρ̆
    concave:   int   # ŝ — neurons that get ρ̂ = -ρ̆(-·)
    saturated: int   # s̃ — neurons that get ρ̃ (saturated piecewise)

    def total(self) -> int:
        return self.convex + self.concave + self.saturated

    def __post_init__(self) -> None:
        for f in ("convex", "concave", "saturated"):
            v = getattr(self, f)
            if v < 0:
                raise ValueError(f"{f} must be ≥ 0; got {v}")

    @classmethod
    def equal_thirds(cls, out_features: int) -> "ActivationSplit":
        """Default split. Remainder (when out_features % 3 != 0)
        is added to `convex`, then `concave`, in that order.
        """
        ...
```

`MonoLinearConfig` (already in `mononet.core.config`) gains a new required field `activation_split: ActivationSplit`, with `from_dict` / `to_dict` roundtripping updated.

### 3.2 New: extend `_KNOWN_ACTIVATIONS`

Current set is `{relu, tanh, sigmoid, elu}`. Add **`selu`** and **`gelu`** (the paper experiments with these). `tanh` and `sigmoid` stay supported — they are bounded and monotone, valid `ρ̆ ∈ Ă` once normalized.

For each named activation we provide three "computed" siblings in `mononet.core.reference`:

```python
def base_activation(name: ActivationName, x: np.ndarray) -> np.ndarray: ...   # ρ̆
def concave_reflection(name: ActivationName, x: np.ndarray) -> np.ndarray: ...  # ρ̂
def saturated_piecewise(name: ActivationName, x: np.ndarray) -> np.ndarray: ... # ρ̃
```

These three are the **arithmetic ground truth** for what every backend must compute when emitting the three "rows" of `ρ_s`. Backends do not duplicate the math; they just route to their framework's `relu` / `elu` / etc. and combine.

### 3.3 Fill in `mononet.core.reference.monotonic_dense`

The signature is already locked. Body:

```python
def monotonic_dense(
    x:              npt.NDArray[np.floating],   # (B, n)
    weights:        npt.NDArray[np.floating],   # (n, m)
    bias:           npt.NDArray[np.floating],   # (m,)
    mask:           MonotonicityMask,           # length n
    activation:     ActivationSpec,             # ρ̆ name
    activation_split: ActivationSplit,          # (s̆, ŝ, s̃), sum = m
) -> npt.NDArray[np.floating]:                  # (B, m)
    # 1. W'  = |W|_t                            -- masked weights
    # 2. h   = x @ W' + b
    # 3. y[:, :s̆]              = ρ̆(h[:, :s̆])
    #    y[:, s̆:s̆+ŝ]          = ρ̂(h[:, s̆:s̆+ŝ])
    #    y[:, s̆+ŝ:]            = ρ̃(h[:, s̆+ŝ:])
    return y
```

`monotonic_mlp` (also already stubbed) chains `K` calls; first layer uses the user's mask, deeper layers force mask = `+1` everywhere (Fig. 4 in the paper).

### 3.4 New: `monotonic_feature_block` (Type 2)

```python
def monotonic_feature_block(
    monotonic_inputs:      npt.NDArray[np.floating],  # (B, n_mono)
    nonmonotonic_features: npt.NDArray[np.floating],  # (B, n_nm_features)
                                                       # ← caller's prior arbitrary net output
    per_feature_weights:   list[npt.NDArray[np.floating]],
    per_feature_biases:    list[npt.NDArray[np.floating]],
    ...
) -> npt.NDArray[np.floating]:
    """One mono-dense block per monotonic input, concat with
    non-monotonic feature vector, then standard mono-MLP tail.
    Matches Fig. 5 in the paper.
    """
```

Each monotonic input feeds its own small mono-dense block; outputs are concatenated with the (pre-computed) non-monotonic feature vector; the tail is a `monotonic_mlp` with all-`+1` masks. The non-monotonic feature extractor itself is **out of scope** — users plug in whatever they want (CNN, transformer, identity) and pass the output to the block.

## 4. Per-backend implementation contract

For each of `torch`, `jax`, `keras`, the trio:

```
mononet/<backend>/
  _kernels.py    # pure functional ops, takes/returns native tensors
  layers.py      # public Module/Layer wrappers around _kernels
  models.py      # public composed models: MonoMLP, MonoFeatureBlock
```

### 4.1 `_kernels.py`

One function per primitive, matching the NumPy reference signature one-for-one (with native-tensor types). For PyTorch:

```python
def monotonic_dense_kernel(
    x:                torch.Tensor,
    weight:           torch.Tensor,        # (in, out)
    bias:             torch.Tensor,        # (out,)
    mask:             torch.Tensor,        # int8, (in,), values in {-1, 0, 1}
    activation_name:  str,                 # "relu" | "elu" | ...
    s_convex:         int,
    s_concave:        int,
    s_saturated:      int,
) -> torch.Tensor:
    ...
```

Kernels are stateless; all parameters come in via arguments. This is what the equivalence harness validates.

Identical-shape signatures for JAX and Keras (`jnp.ndarray` and `keras.KerasTensor` respectively).

### 4.2 `layers.py`

The user-visible class:

```python
class MonoLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec   | str = "elu",
        activation_split: ActivationSplit | None = None,
        init: InitSpec | None = None,
        bias: bool = True,
    ) -> None: ...
    # weight: nn.Parameter of shape (in, out), unconstrained
    # bias:   nn.Parameter of shape (out,), or None
    # forward(x): calls _kernels.monotonic_dense_kernel(...)
```

- `activation_split=None` resolves to `ActivationSplit.equal_thirds(out_features)`.
- Accepts either an `ActivationSpec` object or a plain string (auto-wrap).
- Backend equivalents identical apart from `nn.Module` → `nnx.Module` / `keras.Layer`.

Each layer also accepts an `__init__` overload `from_config(cls, cfg: MonoLinearConfig) -> Self` for benchmark reproducibility (configs serialize to JSON, see parent spec §3).

### 4.3 `models.py`

```python
class MonoMLP(nn.Module):
    """Fig. 4 architecture — input mask + all-positive masks on deeper layers."""
    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        out_features: int,
        monotonicity: MonotonicityMask,
        activation: ActivationSpec | str = "elu",
        activation_split: ActivationSplit | None = None,    # applied to every hidden layer
        output_activation: ActivationSpec | str | None = None,
        init: InitSpec | None = None,
    ) -> None: ...

class MonoFeatureBlock(nn.Module):
    """Fig. 5 architecture — per-monotonic-input blocks, then mono-MLP tail.

    Users supply the non-monotonic feature extractor externally (any
    nn.Module); MonoFeatureBlock concatenates its output with the
    per-monotonic-feature embeddings before the mono-MLP tail.
    """
    def __init__(
        self,
        per_monotonic_widths: Sequence[int],   # one width per monotonic input
        nonmonotonic_features: int,            # size of the prior net's output vector
        hidden_features: Sequence[int],
        out_features: int,
        monotonicity: MonotonicityMask,        # only constrains the per-feature blocks
        activation: ActivationSpec | str = "elu",
        activation_split: ActivationSplit | None = None,
        output_activation: ActivationSpec | str | None = None,
        init: InitSpec | None = None,
    ) -> None: ...

    def forward(
        self,
        monotonic_inputs:      torch.Tensor,    # (B, n_mono)
        nonmonotonic_features: torch.Tensor,    # (B, nonmonotonic_features)
    ) -> torch.Tensor: ...
```

Identical class shapes in JAX (Flax NNX) and Keras (with `MonoDense` instead of `MonoLinear`).

## 5. Cross-backend equivalence (`tests/equivalence/`)

### 5.1 Test vector format

`tests/equivalence/cases/mono_linear/*.json` — committed test vectors. Each case:

```json
{
  "name": "mono_linear-2x3-elu-1-1-1-mask++0",
  "shapes":           { "batch": 4, "in_features": 2, "out_features": 3 },
  "monotonicity":     [1, 1, 0],
  "activation":       "elu",
  "activation_split": [1, 1, 1],
  "init_seed":        17,
  "dtype":            "float32",
  "expected_output":  [[...], [...], [...], [...]],   // 4 rows × 3 cols
  "expected_grads":   { "weight": [...], "bias": [...] },
  "tol":              { "atol": 1e-5, "rtol": 1e-5 }
}
```

These are **generated once** by running the NumPy reference (which is the source of truth) over a grid of `(shape, mask, activation, split, seed, dtype)` combinations, then committed to git. CI never regenerates them — flaky-seed-free guarantee.

### 5.2 Grid coverage

Generated test cases must cover at minimum (the cartesian product is pruned to ~40 cases):

| Axis | Values |
|---|---|
| `(batch, in, out)` | `(1,1,1)`, `(4,2,3)`, `(8,7,12)`, `(2,16,1)`, `(3,5,11)` |
| `monotonicity` | all-`+1`, all-`-1`, all-`0`, mixed `{+1,-1,0}`, `+1`-only |
| `activation` | `relu`, `elu`, `selu`, `gelu`, `tanh`, `sigmoid` |
| `activation_split` | `(m,0,0)` convex-only, `(0,m,0)` concave-only, `(0,0,m)` saturated-only, equal-thirds, `(1,1,m-2)` |
| `dtype` | `float32`, `float64` |

### 5.3 The test itself

```python
@pytest.mark.parametrize("case", load_cases("mono_linear"))
def test_mono_linear_matches_reference(case: EquivalenceCase, backend_name: str):
    backend = backend_for(backend_name)        # importorskip
    expected = case.expected_output            # already computed via reference
    got = backend.run_mono_linear_kernel(case.inputs())
    np.testing.assert_allclose(got, expected, atol=case.atol, rtol=case.rtol)

@pytest.mark.parametrize("case", load_cases("mono_linear"))
def test_mono_linear_gradients_match(case, backend_name: str):
    # Backend autograd vs NumPy finite differences (computed once at
    # vector-generation time and stored in `expected_grads`).
    ...
```

`backend_name` is parametrized by an indirect fixture over `{torch, jax, keras}`, automatically skipping any backend not installed in the current CI job — matches the per-backend matrix from the parent spec §6.

### 5.4 Property tests (Hypothesis)

A small Hypothesis-based test asserts the monotonicity *property* itself, framework-by-framework: for any input pair `(x, x′)` with `x ≤ x′` componentwise on `+1`-positions and `x ≥ x′` componentwise on `-1`-positions, the layer's output is `≤ y′` componentwise. Hypothesis is already in the `dev` group of the parent spec.

This is a complement to the reference-equivalence tests: they prove the kernel matches NumPy; the property test proves NumPy actually implements the monotonicity guarantee.

## 6. Backend-specific notes

### 6.1 PyTorch

- `MonoLinear(nn.Module)` stores `weight: nn.Parameter` of shape `(in_features, out_features)` (column-vectors-per-output is the paper convention; matches `torch.nn.Linear`'s shape modulo transpose). The kernel internally applies `|.|_t` on a fresh tensor — no in-place mutation of the parameter, so the optimizer continues to update unconstrained weights.
- Optional `torch.compile` smoke test (skipped on older versions) verifies the kernel survives compilation. Not required for parity.

### 6.2 JAX (Flax NNX)

- `MonoLinear(nnx.Module)`. Use `nnx.Variable` for `weight` and `bias`.
- Kernel is `jax.jit`-able by construction (pure functional, no Python control flow over arrays). The activation-split routing uses `jnp.concatenate` of three slices; no branching on tensor values.
- An explicit smoke test asserts `jax.grad(loss)` and `jax.vmap` both work on `MonoLinear.__call__`.

### 6.3 Keras 3

- `MonoDense(keras.Layer)` uses `keras.ops` only. CI runs with `KERAS_BACKEND=jax` per parent spec §3 / §6.
- An additional smoke test (one CI job) runs with `KERAS_BACKEND=torch` to verify the `keras.ops`-based kernel is backend-agnostic.

## 7. Public-API surface this sub-project locks

After Sub-project A lands:

```python
from mononet.core.types  import MonotonicityMask, ActivationSpec, ActivationSplit, InitSpec
from mononet.core.config import MonoLinearConfig
from mononet.core.reference import (
    monotonic_dense, monotonic_mlp, monotonic_feature_block,
    base_activation, concave_reflection, saturated_piecewise,
)

from mononet.torch import MonoLinear, MonoMLP, MonoFeatureBlock
from mononet.jax   import MonoLinear, MonoMLP, MonoFeatureBlock
from mononet.keras import MonoDense,  MonoMLP, MonoFeatureBlock
```

This is the **frozen** surface that Sub-projects B (benchmarks), C (extended benchmarks), and D (flows) build on. Breaking changes after this point require a major version bump.

## 8. Default choices (what we ship out of the box)

| Choice | Default | Why |
|---|---|---|
| Default `ρ̆` (base activation) | `"elu"` | Paper's preferred non-saturated convex activation; gradient continuous; gave best Table 1 results in the paper's own ablation. |
| Default `activation_split` for `out_features = m` | `ActivationSplit.equal_thirds(m)` | Paper-recommended starting point. Resolves to `(⌈m/3⌉, ⌈(m - ⌈m/3⌉)/2⌉, rest)` with the convex bucket greedy. |
| Default `init` scheme | `"glorot_uniform"` | Matches the existing `InitSpec` default. ELU + glorot is a known-good combo; no need for SELU-style normalised init. |
| Whether `bias` defaults to `True` | `True` | Matches `nn.Linear`. |
| Deeper-layer mask | All-`+1`, set internally by `MonoMLP` | Required by the paper construction; users should not be able to break monotonicity by accident. |
| Output activation | `None` (identity) | Users pick task-specific tail (sigmoid for binary classification, softmax for multi-class, identity for regression). |

## 9. Open items

These do not block writing the implementation plan; they only need a decision *during* the plan:

- Should `MonoLinear` accept `activation_split` as a tuple `(s̆, ŝ, s̃)` for convenience, in addition to the dataclass? *Lean yes, with an internal coerce-to-dataclass.*
- For Type 2 (`MonoFeatureBlock`): one nonmonotonic feature extractor for all features, or per-feature? The paper shows one global extractor (Fig. 5). *Match the paper: one extractor, output concatenated.*
- Equivalence harness: regenerate test vectors when the reference implementation changes, or also store the reference Python code's git hash next to them? *Store hash + add a `regenerate-cases.py` script under `tools/`.*
- For Keras 3: support TF backend in the smoke matrix, or just JAX + Torch backends? *Add TF backend smoke run only if it doesn't push CI cost meaningfully; otherwise drop.*

## 10. What is intentionally NOT in this sub-project

- **Benchmarks**. Even toy comparisons. Sub-project B owns dataset loaders, training scripts, and the Tables 1 / 2 reproduction.
- **Invertibility / strict monotonicity**. The construction here uses `|W|_t`, which is monotone *non-strict* — there is no guarantee of injectivity. Sub-project D introduces a strictly-positive weight parameterization (e.g. `softplus(W) + ε`) wrapped in `ScalarMonoBijector` that is provably injective on ℝ. Putting that here would over-engineer the base layer for the common case.
- **Lean proofs of monotonicity**. The Hypothesis property test gives empirical evidence; Lemma 1 / Lemma 2 of the paper are formally proven in Sub-project E.
- **Saving / loading**. Each framework's native serialization (`state_dict`, `nnx.split`, `keras.saving`) is expected to "just work" because layers are vanilla `Module`/`Layer` subclasses. We test this with a single round-trip smoke test per backend; we do not invent a cross-backend save format.
