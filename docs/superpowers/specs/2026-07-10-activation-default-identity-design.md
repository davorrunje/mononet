# Default activation → `identity` (and mandatory activation for residual blocks)

Status: Approved (design)

## 1. Problem

`MonoLinear` / `MonoDense` and their configs default `activation` to `"relu"`.
This diverges from every native dense layer — `torch.nn.Linear` has no
activation and `keras.layers.Dense` defaults `activation=None` (linear) — and
it caused a real correctness bug: in PR #72 the benchmark read-out head was
built without an explicit activation, silently inherited the `relu` default,
and in `absolute` mode forced the pre-sigmoid `≥ 0`, collapsing every binary
classifier to the base rate (heart 0.21, compas 0.44, loan 0.49). #72 fixed the
symptom by adding a first-class `identity` activation and using it explicitly
for the head, and recorded the follow-up decision to flip the *default* — this
spec.

The default is insidious because it is a **silent** hazard: code keeps running,
but a layer that should be a linear read-out becomes nonlinear.

## 2. Goal & constraints

- Make the default match native-framework convention (`identity`/linear) so a
  `MonoLinear` behaves like a `Linear`, and the read-out footgun is impossible.
- For residual blocks, make the activation an **explicit, mandatory** choice
  rather than a default — a residual block whose `F` is silently linear is a
  different, equally undesirable footgun.
- Breaking change is acceptable: the package is at `0.0.0a0` (pre-release).
- Do not touch the math (kernels), the cross-backend equivalence harness, the
  positivity gates, `convex_fraction`, or `MonoInput`.

## 3. Design

### 3a. Leaf dense layers default to `identity`

`MonoLinear` (PyTorch, JAX), `MonoDense` (Keras), and `MonoConfig` change their
`activation` default from `"relu"` to `"identity"`. A bare `MonoLinear(d, h)` is
then a monotone **linear** map (`|W|·x + b` in `absolute`; the switch-mode
degenerate linear form otherwise); nonlinearity is opt-in via `activation=`.

The NumPy reference `monotonic_dense` default flips to match, so the three
backends and the reference stay mutually consistent. Committed equivalence
vectors set `activation` explicitly, so they are unaffected.

Consequence, stated plainly: mononet applies the activation **inside** the layer
(it is a parameter, not a separate module inserted between layers). So a bare
`Sequential(MonoLinear, MonoLinear, …)` with the new default is linear
end-to-end — the user must set `activation=` on the hidden layers to get a
nonlinear network. This matches `nn.Linear`/`Dense` (where you likewise add
nonlinearity explicitly) and is the documented convention. There is no composed
model class to attach a guardrail to, so the convention stands on documentation.

### 3b. Residual blocks require an explicit activation (Option A)

`MonoResidual` (all three backends): `activation` becomes
`ActivationSpec | str | None = None`, validated in `__init__`:

- `F is None and activation is None` → `ValueError("activation is required when
  F is not provided; the default F would otherwise be linear")`.
- `F is not None and activation is not None` → `ValueError("pass either F or
  activation, not both")` — mirrors the existing `F`/`sub_depth` mutual
  exclusion (`activation` only ever configures the default `F`).

So: no silently-linear default `F`, and no ignored `activation` alongside a
custom `F`.

`MonoResidualConfig` has no `F` field (it only describes the serializable
default block), so it simply **requires** `activation`. Mechanism:
`activation: ActivationSpec = field(kw_only=True)` (Python ≥3.11 per-field
keyword-only) — keeps `units`/`mode` positional while allowing a required field
after defaulted ones. `from_dict` already passes `activation` by keyword, so the
JSON round-trip is unchanged; constructing the config without `activation`
raises `TypeError`.

### 3c. Untouched

Kernels, the cross-backend equivalence harness, `convex_fraction`, the residual
gates (`alpha_gate`/`beta_gate`), and `MonoInput`.

## 4. Concrete changes

- `mononet/core/config.py` — `MonoConfig.activation` default `relu`→`identity`;
  `MonoResidualConfig.activation` becomes required (`field(kw_only=True)`, no
  default).
- `mononet/core/reference.py` — `monotonic_dense` activation default
  `relu`→`identity`.
- `mononet/torch/layers.py`, `mononet/jax/layers.py`, `mononet/keras/layers.py`:
  - `MonoLinear`/`MonoDense`: `activation` default `relu`→`identity`.
  - `MonoResidual`: `activation` default → `None` + the Option-A validation.
- Docstrings: update `:param activation:` defaults on all touched
  layers/configs.
- `CHANGELOG.md` — a **Breaking changes** entry (see §5).
- Docs/examples audit (see §6).

## 5. Migration

Two break kinds:

- **Hard / loud** — `MonoResidual` and `MonoResidualConfig` now require
  `activation`; calls that omit it raise immediately. Desirable: fails fast.
- **Silent / behavioral** — leaf default `relu`→`identity`; existing code runs
  but is now linear.

No deprecation shim: at `0.0.0a0` there is no stability promise, and we cannot
distinguish "relied on the implicit default" from an explicit `activation="relu"`.
Mitigation is a prominent CHANGELOG **Breaking changes** entry with the one-line
recovery: *pass `activation="relu"` explicitly to restore the previous
behavior*. Version bump (e.g. `a1`) is left to the maintainer's release cadence.

## 6. Testing & docs

Regression tests (new, per backend where relevant):

- A bare `MonoLinear`/`MonoDense` under the default activation is affine in `x`
  (`f(ax₁+bx₂) == a·f(x₁)+b·f(x₂)` up to tolerance) — confirms the `identity`
  default across PyTorch/JAX/Keras.
- `MonoResidual`: raises when `F is None` and `activation` omitted; raises when
  both `F` and `activation` are passed; constructs and runs when `activation` is
  given.
- `MonoResidualConfig()` without `activation` raises `TypeError`.
- Run the existing cross-backend equivalence and monotonicity property suites
  unchanged (committed vectors are explicit → expected green).

Docs/examples:

- Audit README, `docs/guides/{pytorch,jax,keras}`, and `docs/concepts/*` for
  examples that stack `MonoLinear`/`MonoDense` relying on the implicit `relu`;
  add explicit `activation=` so rendered examples remain nonlinear networks.
- The strict Sphinx docs build must stay green.

Benchmarks: `benchmarks/_common/model_builder.py` already passes `activation`
explicitly (head is `identity`; hidden layers use `cfg.activation`), and
benchmark configs default to `"elu"`, so no benchmark code relies on the package
default. Run the benchmark unit tests to confirm.

## 7. Out of scope

- `convex_fraction`, the positivity gates, `MonoInput`.
- New composed model classes / stack-level guardrails.
- Refreshing the stale residual flavor-comparison numbers (separate follow-up).
- The version bump itself (release-pipeline concern).

## 8. Open items

- Whether to bump the version to `0.0.0a1` as part of this change or defer to
  the next release — maintainer decision, noted in the CHANGELOG either way.
