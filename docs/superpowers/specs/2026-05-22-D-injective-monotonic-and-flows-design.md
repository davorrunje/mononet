# Sub-project D — Injective monotonic blocks and normalizing flows

**Date:** 2026-05-22
**Author:** Davor Runje
**Status:** Draft (brainstorming output); pending user review.
**Parent spec:** [`2026-05-21-mononet-package-design.md`](2026-05-21-mononet-package-design.md)
**Depends on:** [Sub-project A](2026-05-22-A-core-algorithm-and-backends-design.md) (locked public API)
**Primary references:**
- Runje & Shankaranarayana, *Constrained Monotonic Neural Networks*, ICML 2023 — <https://arxiv.org/abs/2205.11775>
- Wehenkel & Louppe, *Unconstrained Monotonic Neural Networks*, NeurIPS 2019 — <https://arxiv.org/abs/1908.05164>
- Durkan et al., *Neural Spline Flows*, NeurIPS 2019
- Papamakarios et al., *Normalizing Flows for Probabilistic Modeling and Inference*, JMLR 2021

## 1. Goals & non-goals

### Goals

- Build a **strictly monotonic** scalar primitive on top of Sub-project A's CMFCL — a function ℝ → ℝ that is provably increasing (not just non-decreasing) so it can serve as the building block for invertible bijectors.
- Provide `ScalarMonoBijector`: a 1-D strictly-monotonic bijection ℝ → ℝ exposing `forward`, `inverse`, and `log_det_jacobian`.
- Build `MonoCouplingFlow` (RealNVP-style coupling with `ScalarMonoBijector` as the transformer) and `MonoAutoregressiveFlow` (MAF/IAF-style with `ScalarMonoBijector` conditioned on previous coordinates).
- Position the library against UMNN (Wehenkel & Louppe 2019), which uses *integration of a positive net* for the same purpose. Provide an `IntegralMonoBijector` that is API-compatible, so users can A/B-test the two approaches on the same flow.
- Benchmark on the standard density-estimation suite: POWER, GAS, HEPMASS, MINIBOONE, BSDS300, MNIST/CIFAR likelihood.
- Initial release in **one backend** to keep scope bounded; recommended PyTorch (largest flow-library ecosystem).

### Non-goals

- No re-derivation of monotonic-flow theory. We use the existing literature.
- No conditional flows, variational inference, or generative-modeling pipelines beyond density estimation. Sampling from a flow comes for free with the bijector API; we expose it but do not build full VI tooling.
- No discrete flows, continuous flows (Neural ODE), or score-based diffusion adjacency. Out of scope.
- No multi-backend release in v1 of this sub-project. JAX and Keras ports become candidate follow-up sub-projects after PyTorch lands.
- No application to images at full resolution. We stop at MNIST/CIFAR density-estimation tables; image generation is out.

## 2. The injectivity gap

Sub-project A's `MonoLinear` uses `|W|_t`, which gives weakly-monotonic outputs:

```
∂y_j / ∂x_i  ≥ 0   when t_i = +1     (only ≥, not >)
```

Equality occurs whenever `W_{j,i}` is exactly zero or the post-activation gradient is zero (e.g. ReLU on the negative side). A flow needs strict monotonicity:

```
∂T(x) / ∂x  > 0     for all x   →   T is a bijection of ℝ
```

Two ways to upgrade:

### Option 1 — Strictly-positive weight parameterization

Replace `|W|` with a strictly-positive parameterization:

- `softplus(W) + ε`, ε ∈ (0, ∞)
- `exp(W)` (no ε needed; range is (0, ∞) for W ∈ ℝ)
- `|W| + ε` with ε > 0

Combined with an activation whose derivative is strictly positive on all of ℝ (e.g. `softplus`, `tanh`, `sigmoid`, smoothed ELU like CELU with α > 0), the resulting scalar function is strictly increasing.

**Tradeoff**: lose the literal `|·|_t` construction; the universal-approximation proof (Sub-project A) does not directly carry over because Theorem 7 was proven for `|W|_t`, not `softplus(W)`. We will state the strictly-positive parameterization as a separate construction with its own (much shorter) approximation argument, citing standard monotone-network universality (Daniels & Velikova 2010).

### Option 2 — Integral construction (UMNN-style)

Define `T(x) = T(0) + ∫₀ˣ g(u; θ) du` where `g` is a strictly-positive neural network. With `g = softplus(ρ(MLP(u)))` for any `ρ ∈ Ă`, `T` is C¹ and strictly increasing. Inverse and `log_det_jacobian = log g(x)` are direct.

**Tradeoff**: forward pass requires numerical integration (Clenshaw-Curtis, fixed nodes); training pays an extra ~16-32 function evaluations per scalar transform. UMNN paper shows this is competitive with affine flows.

### Decision

**Ship both.** Option 1 is the `mononet`-native bijector, with smaller per-step cost. Option 2 (`IntegralMonoBijector`) is provided as a same-API alternative so users — and our own benchmark — can compare. The integral version's strict-positive net is itself a `MonoMLP` from Sub-project A, so we reuse maximum infrastructure.

## 3. New module: `mononet.flows`

This is a new top-level submodule. It depends on `mononet.torch` only in v1.

```
mononet/
└── flows/
    ├── __init__.py          # public exports
    ├── _common.py           # Bijector base class, log-prob plumbing
    ├── scalar.py            # ScalarMonoBijector, IntegralMonoBijector
    ├── coupling.py          # MonoCouplingFlow
    ├── autoregressive.py    # MonoAutoregressiveFlow (MAF) + inverse (IAF)
    └── compose.py           # FlowSequence (stack of bijectors)
```

`pyproject.toml` gets a new extra:

```toml
flows = ["mononet[torch]", "torch>=2.4"]
```

Pure-PyTorch in v1; no `torch.distributions` dependency beyond the standard `torch.distributions.Normal` for base densities.

## 4. The `Bijector` interface

```python
class Bijector(nn.Module):
    """Differentiable invertible mapping ℝ^d → ℝ^d."""

    event_shape: tuple[int, ...]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """:returns: (y, log|det J_forward|)"""
        ...

    def inverse(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """:returns: (x, log|det J_inverse|)"""
        ...
```

Following `distrax` / `nflows` conventions. `log_det_jacobian` is bundled with the transformation because computing it separately would re-traverse the same forward pass.

A `FlowSequence(Bijector)` wraps a list of bijectors and accumulates log-dets.

## 5. `ScalarMonoBijector`

```python
class ScalarMonoBijector(Bijector):
    """A strictly monotonic 1-D map ℝ → ℝ built from MonoMLP with a
    strictly-positive weight parameterization."""

    def __init__(
        self,
        hidden_features: Sequence[int] = (32, 32),
        positive_param:  Literal["softplus", "exp", "abs_plus_eps"] = "softplus",
        eps:             float = 1e-3,
        base_activation: str = "softplus",
        conditioner_features: int = 0,        # 0 → unconditional; >0 → take a context vector
    ): ...

    def forward(self, x, *, context=None) -> tuple[torch.Tensor, torch.Tensor]: ...
    def inverse(self, y, *, context=None) -> tuple[torch.Tensor, torch.Tensor]: ...
```

**Inverse** is computed by bisection (default) or by `torch.linalg.solve` for the linear-output last-layer case (rarely useful in practice). Bisection runs to a configurable absolute tolerance (default `1e-6`) using a fixed number of iterations (default `30`); we expose `max_iters` and `tol` as constructor kwargs.

**Why bisection over Newton**: bisection is unconditionally convergent for a strictly-increasing scalar function; Newton converges faster but requires bracket safety. The benchmark in §8 measures the per-step cost — if bisection turns out to dominate training time we add Newton with a Brent fallback.

**`log_det_jacobian`** is just `log(∂T/∂x)`, which is `log(weight chain product through positive activations)`. PyTorch autograd computes it; we use `torch.autograd.grad` once per forward pass with `create_graph=True` for second-order gradients during training.

## 6. `IntegralMonoBijector` (UMNN-style)

```python
class IntegralMonoBijector(Bijector):
    """T(x) = T(0) + ∫_0^x softplus(g(u; θ)) du, with g a MonoMLP.
    Implements Wehenkel & Louppe 2019 in mononet."""

    def __init__(
        self,
        hidden_features: Sequence[int] = (32, 32),
        n_integration_points: int = 32,
        integration: Literal["clenshaw_curtis", "gauss_legendre"] = "clenshaw_curtis",
        conditioner_features: int = 0,
    ): ...
```

Implementation reuses `mononet.torch.MonoMLP` for `g`. Integration nodes/weights are precomputed constants; the forward pass is one vectorized MLP call over the integration grid plus a weighted sum. Inverse uses bisection on `T(x) - y = 0`, same as the scalar bijector.

## 7. Coupling flows

`MonoCouplingFlow` is a RealNVP-style coupling layer where the affine transform of the second half is replaced by a `ScalarMonoBijector` (or `IntegralMonoBijector`) applied element-wise, conditioned on the first half via an unconstrained MLP:

```python
class MonoCouplingFlow(Bijector):
    """RealNVP coupling: split (x_a, x_b);  y_a = x_a;  y_b = T(x_b ; context=h(x_a)).
    T is per-coordinate ScalarMonoBijector; h is an unconstrained MLP."""

    def __init__(
        self,
        event_dim: int,
        split:     Literal["alternate", "first_half"] = "alternate",
        bijector_factory: Callable[[int], ScalarMonoBijector] | None = None,
        conditioner_features: Sequence[int] = (64, 64),
    ): ...
```

`MonoAutoregressiveFlow` is analogous to MAF: `y_i = T(x_i; context = h(x_{<i}))`, with `h` a masked MLP. Standard autoregressive trick — fast for log-likelihood, sequential for sampling.

## 8. Benchmark: UMNN replication suite

We replicate Table 1 from Wehenkel & Louppe 2019 — density estimation on five UCI tabular benchmarks:

| Dataset | d | n_train | n_test | Source |
|---|---|---|---|---|
| POWER | 6 | 1.65M | 204k | UCI |
| GAS | 8 | 853k | 105k | UCI |
| HEPMASS | 21 | 315k | 174k | UCI |
| MINIBOONE | 43 | 29k | 3.6k | UCI |
| BSDS300 | 63 | 1M | 250k | Berkeley Segmentation Dataset |

Loaders follow Sub-project B's `DatasetBundle` contract (modified for unsupervised density-estimation; no `y`).

Comparators on the same harness:

| Method | Where it comes from |
|---|---|
| `mononet ScalarMonoBijector` (ours) | This sub-project. |
| `mononet IntegralMonoBijector` (UMNN reimplementation) | This sub-project. |
| MAF (Papamakarios 2017) | Re-implement with affine transformer, ~100 LOC. |
| NSF rational-quadratic (Durkan 2019) | Quote published number; only re-run if a clean PyTorch port is small. |
| Real NVP affine | Re-implement, ~80 LOC. |

Headline plot: NLL test-set per dataset, three "monotonic-flow" rows × five datasets, with our scalar vs. integral bijector compared at matched parameter budgets.

## 9. The injectivity property test

A Hypothesis-based test asserts:

- For `ScalarMonoBijector` with `positive_param ∈ {"softplus", "exp"}`, `forward(x_2) - forward(x_1)` has the same sign as `x_2 - x_1` for all sampled pairs.
- `bijector.inverse(bijector.forward(x)[0])[0] ≈ x` within `atol=1e-5` on `float32`.
- `log_det_jacobian` computed analytically matches `torch.autograd.functional.jacobian(forward, x).log().sum()` for low dimensions.

These tests are not the *formal proof of injectivity* (that's Sub-project E — and likely a follow-up if the formalization stops at the CMNN paper's Theorem 7). They are the empirical guard.

## 10. Public API after Sub-project D

```python
from mononet.flows import (
    Bijector,
    FlowSequence,
    ScalarMonoBijector,
    IntegralMonoBijector,
    MonoCouplingFlow,
    MonoAutoregressiveFlow,
)
```

## 11. Documentation

A new docs section "Guides → Normalizing flows" walks through:

1. The strict-monotonicity gap and the two parameterizations (this spec §2).
2. Building a 2-D toy density model with `MonoCouplingFlow`.
3. The UMNN density-estimation reproduction (link to notebook).
4. When to prefer scalar vs. integral bijectors (recommendation: scalar for moderate `d`, integral for very smooth one-dimensional transforms where the extra fevals are tolerable).

## 12. Open items

- **Strict-positive parameterization default.** `softplus(W) + ε` vs. `exp(W)`. `softplus` keeps gradients gentler; `exp` is closer to UMNN's integrand parameterization. Recommendation: `softplus` default, expose `exp` as opt-in.
- **Bisection vs. Newton.** Benchmark both during implementation. If Newton is materially faster and we can prove safety for our parameterization, switch default.
- **JAX / Keras ports.** Defer to a successor spec. The JAX port in particular benefits from `jax.lax.while_loop` for the inverse, which would change ergonomics.
- **Conditional flows.** Should `ScalarMonoBijector` accept a `context` tensor in v1? Lean yes; the `conditioner_features=0` default keeps the unconditional case clean.
- **Distribution wrappers.** Do we wrap a flow + base distribution into a `Flow(nn.Module)` with `log_prob` / `sample`? Lean yes — it's a 30-line wrapper that makes the public examples cleaner.
- **Patent scope.** US 11,551,063 covers the constrained-weights-with-three-activations construction. The strictly-positive-weights variant for flows is *related* but materially different in parameterization and intended use. Confirm with counsel before publishing benchmarks that present `mononet`'s flow as a derivative work of the patent.

## 13. What is intentionally NOT in this sub-project

- **Theorem-prover formalization of injectivity.** Sub-project E may or may not include it; we treat that as a separate question.
- **Wholesale replacement of UMNN.** Our `IntegralMonoBijector` is a port for benchmark-parity reasons. We do not claim it improves on UMNN; we want a single repo where readers can compare both monotonic-flow constructions side by side.
- **Multi-backend symmetry.** PyTorch-only in v1 of this sub-project. Asymmetric API surface across `mononet.torch`/`jax`/`keras` is acceptable here because flows are a downstream specialty, not a core primitive.
- **Training-loop / dataset wrappers.** Users plug bijectors into their own training code (PyTorch Lightning, etc.).
