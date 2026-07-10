# Default `mode` → `absolute`, `Mode`-typed param, and tested mixed-feature examples

Status: Approved (design)

Follow-up to PR #75 (default activation → identity). Branch: `feat/mode-default-absolute`.

## 1. Problem / motivation

Three items deferred from the #75 review:

1. The layer default `mode` is `"switch"`, but `absolute` (the base `|W|`
   construction from arXiv:2205.11775) is the more natural default and now
   trains at moderate/skip depth thanks to the static init (#66) and
   `MonoResidual.sub_depth` (#67). Make `absolute` the default.
2. The layer `mode` parameter is a bare `str`; it should be the `Mode` literal
   (`Literal["switch","absolute"]`, already defined in `core/config.py`) so
   unknown modes are rejected at type-check time (parallels the `ActivationName`
   tightening in #75).
3. The docs examples are shallow, `switch`-mode, single-feature-type stacks.
   Rework them into a realistic **mixed-feature** network (monotone features
   with mixed +1/−1 directions **plus** non-monotone features embedded through a
   plain MLP), in `absolute` mode, using both `MonoLinear` and `MonoResidual` —
   and make the example code **tested and coverage-included** so it cannot rot.

## 2. Goals & constraints

- Breaking change is acceptable (`0.0.0a0`), with a `CHANGELOG` recovery note.
- Do not touch kernels, the cross-backend equivalence harness, `convex_fraction`,
  the residual gates, `MonoInput`/`MonotonicityMask` semantics, or the
  activation contract from #75.
- `MonotonicityMask` supports only `{-1,+1}` (no unconstrained/0 direction) and
  `MonoInput` sign-flips a vector whose width must equal the mask length — the
  example design must respect this (verified: a wrong-size mask raises).

## 3. Design

### 3a. Default `mode` → `absolute`

Flip the default `mode` from `"switch"` to `"absolute"` in:

- `mononet/core/config.py` — `MonoConfig.mode`, `MonoResidualConfig.mode`.
- `mononet/{torch,jax,keras}/layers.py` — `MonoLinear`/`MonoDense`/`MonoResidual`.
- `mononet/core/reference.py` — `monotonic_dense` default (keeps the reference
  consistent with the layers).

`absolute` with no explicit `init` already selects the static
`absolute_init_params` init (#66), so bare layers train. Blast radius is small:
source, `benchmarks/`, and the test suite all pass `mode` explicitly (audited),
so only code relying on the *implicit* `switch` default changes behaviour.

### 3b. `mode` typed as `Mode`

Layer `mode: str` → `mode: Mode` on all three backends' `MonoLinear`/`MonoDense`/
`MonoResidual`. `Mode` is imported under `TYPE_CHECKING` (annotation-only). The
kernels and `reference.monotonic_dense` keep `mode: str` (internal ground truth,
not the public layer API). Benchmark helpers that pass a bare `str` `mode` into
the tightened layer API (`benchmarks/_common/init_diagnostics.py`
`_stack`/`grad_flow`/`trainability`, `benchmarks/deep_init_run.py`) are typed
`Mode`; the benchmark config already uses the literal. Same propagation pattern
as `ActivationName` in #75.

### 3c. Reworked, tested examples

Canonical example: a mixed-feature monotone network, per backend.

- **Source of truth:** `docs/examples/risk_net_torch.py`, `…_jax.py`,
  `…_keras.py` — each a small module defining a `RiskNet` that is:
  - monotone in 3 features via `MonoInput(MonotonicityMask([+1,+1,-1]))`
    (2 non-decreasing, 1 non-increasing),
  - unconstrained in 2 non-monotone features, embedded through two plain FC
    layers (`2→16→8`, framework-native `Linear`/`Dense`),
  - concatenated (`[MonoInput(x_mono), embed(x_free)]` → 11 columns) and passed
    through an `absolute`-mode stack: `MonoLinear(11,64,activation="elu")` →
    2× `MonoResidual(64,64,activation="elu")` → `MonoLinear(64,1)` (identity
    read-out). No explicit `mode=` (absolute is the default).
  - Teaching point documented: the embedding absorbs the non-monotonicity, so
    the composite is monotone in `x_mono` and unconstrained in `x_free`.
- **Docs:** `docs/guides/{pytorch,jax,keras}.md` embed the code via Sphinx
  `literalinclude` (single source of truth, no drift). `README.md` carries the
  torch version inline (GitHub/PyPI markdown cannot `literalinclude`).
- **Tests:** `tests/examples/test_risk_net_{torch,jax,keras}.py` (per backend,
  `importorskip`) import the module, run a forward, and assert monotonicity —
  bump each +1 feature ⇒ output non-decreasing; the −1 feature ⇒ non-increasing
  (as validated during design). Plus a `tests/examples/test_readme_matches.py`
  that asserts the README torch code block is byte-identical to
  `docs/examples/risk_net_torch.py`'s body (README drift guard).
- **Coverage:** `benchmarks` is already in the coverage source
  (`addopts = --cov=mononet --cov=benchmarks`); add `--cov=docs/examples` so
  the example modules are counted too. The tests load the example modules by
  file path (`importlib`), so `docs/` need not become an importable package,
  and the forward pass exercises the module top-to-bottom — an API break trips
  CI.

## 4. Concrete changes

- `mononet/core/config.py`, `mononet/core/reference.py`,
  `mononet/{torch,jax,keras}/layers.py` — default `mode` flip + `Mode` typing
  on the layer params (config/reference already use their own annotations).
- `benchmarks/_common/init_diagnostics.py`, `benchmarks/deep_init_run.py` —
  `mode: str` helper params → `Mode`.
- `docs/examples/risk_net_{torch,jax,keras}.py` — **new** example modules.
- `docs/guides/{pytorch,jax,keras}.md`, `README.md` — swap in the new example
  (`literalinclude` in guides; inline in README).
- `tests/examples/` — **new** monotonicity tests (per backend) + README-match
  test.
- `pyproject.toml` — add `--cov=docs/examples` to
  `[tool.pytest.ini_options].addopts` (`--cov=mononet --cov=benchmarks` already
  present).
- `CHANGELOG.md` — breaking-change entry (see §5).

## 5. Migration

`mode` default `switch`→`absolute` is a **silent behavioural** breaking change
(existing implicit-default code keeps running but computes the `absolute`
construction). No deprecation shim at `0.0.0a0`. `CHANGELOG` **Breaking
changes** entry with the recovery note: pass `mode="switch"` explicitly to
retain the previous behaviour. The `Mode` typing is a compile-time-only tighten
(no runtime behaviour change).

## 6. Testing

- New: default-`mode`-is-`absolute` assertion (`MonoConfig`, bare layer).
- New: per-backend `RiskNet` forward + monotonicity tests (the design-validated
  checks) and the README-match test.
- Unchanged: the cross-backend equivalence and monotonicity suites stay green
  (they set `mode` explicitly).
- `uv run mypy` clean across all backends (incl. the benchmark `Mode`
  propagation); strict `sphinx-build -W` clean (the `literalinclude` targets
  must exist and import cleanly).

## 7. Out of scope

- Kernels, equivalence harness, `convex_fraction`, gates, `MonoInput`/
  `MonotonicityMask` semantics, the #75 activation contract.
- Any new shipped-package model class (`RiskNet` lives under `docs/examples/`,
  not in the `mononet` wheel).
- A 0/unconstrained direction in `MonotonicityMask` (non-monotone features are
  handled by embedding, per §3c).

## 8. Open items

- Whether to bump the version (`a1`) — maintainer/release-cadence decision,
  noted in the `CHANGELOG` either way.
