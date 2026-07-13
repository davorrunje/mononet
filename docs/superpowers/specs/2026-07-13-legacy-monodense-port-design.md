# Legacy MonoDense Port — Design

**Status:** Draft for review
**Date:** 2026-07-13
**Owner:** Davor Runje
**Triggering event:** The new `mononet` package is not API-compatible with the
original [airtai/monotonic-nn](https://github.com/airtai/monotonic-nn)
`MonoDense` layer. The new `MonoLinear`/`MonoDense` uses a two-value monotonicity
mask (`±1`), whereas the original used a three-value `monotonicity_indicator`
(`{-1, 0, 1}`, with `0` = non-monotonic). Existing users of the original package
need a drop-in bridge to migrate onto `mononet`.

## Goal

Provide `mononet.legacy` — a faithful, backend-agnostic reproduction of the
original `airt` `MonoDense` layer and its associated public API — so that code
written against the original `monotonic-nn` package can be ported to `mononet`
with minimal changes. Using any legacy symbol emits a `DeprecationWarning`.

## Non-goals

- Reproducing the original repo's experiments / training / dataset code
  (`airt/keras/experiments.py`). The `mononet` wheel ships **layers only**.
- Reintroducing a hard `tensorflow` dependency. The port is written against
  `keras.ops` and runs under any Keras 3 backend.
- Integrating the legacy layer with the new architecture (`MonotonicityMask`,
  `_kernels.py`, cross-backend equivalence harness). Legacy is a self-contained
  faithful copy, not a participant in the new equivalence tests.
- Changing the new `mononet.keras.MonoDense`. It coexists with the legacy layer
  in a separate namespace.

## Background

The original `MonoDense` (in `airt/_components/mono_dense_layer.py`) subclasses
`tensorflow.keras.layers.Dense` and, at call time:

1. Replaces the kernel `W` with a sign-constrained version driven by the
   `monotonicity_indicator` (`+1` → `|W|`, `-1` → `-|W|`, `0` → `W` unchanged),
   via the `replace_kernel_using_monotonicity_indicator` context manager.
2. Runs the standard `Dense` forward pass with the constrained kernel.
3. Applies a **three-class activation split** (`apply_activations`): the output
   units are partitioned into convex / concave / saturated groups. The split
   sizes come from `activation_weights=(7.0, 7.0, 2.0)` (normalised), unless
   `is_convex`/`is_concave` force an all-convex or all-concave split. The
   saturated activation is synthesized from the base activation via
   `get_saturated_activation`.

The original public surface (the module's `__all__` plus the builder
classmethods) is:

- Layer: `MonoDense`
- Helpers: `get_saturated_activation`, `get_activation_functions`,
  `apply_activations`, `get_monotonicity_indicator`,
  `apply_monotonicity_indicator_to_kernel`,
  `replace_kernel_using_monotonicity_indicator`
- Network builders: `create_type_1`, `create_type_2` (exposed both as
  `MonoDense.create_type_*` classmethods and as `_create_type_*` functions)

This differs from the new `mononet` architecture in two load-bearing ways:

- **Mask domain.** The original allows `0` (non-monotonic) per weight; the new
  `MonotonicityMask` forbids `0` and only accepts `±1`.
- **Activation model.** The original uses the paper's 3-class split with tunable
  `activation_weights`; the new package uses a 2-class convex/concave split
  (`convex_fraction`) or the `switch` mode.

Because of these differences the legacy layer is reproduced verbatim in behavior
rather than mapped onto the new types.

## Design

### Placement & namespace

```
mononet/legacy/
├── __init__.py          # public re-exports + deprecation-warning wiring
└── mono_dense_layer.py  # the port (mirrors the original module name)
```

- Public API: `from mononet.legacy import MonoDense, create_type_1, create_type_2`
  (and the helper functions listed above).
- **Lazy import preserved.** `import mononet` must not import `mononet.legacy`
  (which pulls `keras`). Nothing is added to the top-level `mononet/__init__.py`
  — identical posture to the `torch`/`jax`/`keras` backends.
- `create_type_1` / `create_type_2` are exposed both as module-level functions
  and as `MonoDense.create_type_*` classmethods (delegating to the functions),
  matching the original drop-in surface.
- The new `mononet.keras.MonoDense` and legacy `mononet.legacy.MonoDense` live in
  separate namespaces — no collision.

### Port scope & fidelity (Keras 3, backend-agnostic)

Reproduce the following using `keras.ops` (no `tensorflow` import), numerically
identical to the original and bit-identical under the Keras TF backend:

- **`MonoDense`** — subclass of `keras.layers.Dense` (preserves `kernel`/`bias`
  weight names and Dense config keys for serialization fidelity). Signature
  unchanged:
  `units, *, activation=None, monotonicity_indicator=1, is_convex=False,
  is_concave=False, activation_weights=(7.0, 7.0, 2.0), **kwargs`.
  Keeps the raw `{-1, 0, 1}` indicator semantics. Does **not** use
  `MonotonicityMask`. `get_config` matches the original keys.
- **Helpers** (module-level, same names and signatures):
  `get_saturated_activation`, `get_activation_functions`, `apply_activations`,
  `get_monotonicity_indicator`, `apply_monotonicity_indicator_to_kernel`,
  `replace_kernel_using_monotonicity_indicator`.
- **Builders**: `create_type_1`, `create_type_2`, plus the internal
  `_create_mono_block`, `_prepare_mono_input_n_param`,
  `_check_convexity_params`. Ported using `keras.layers.Concatenate`,
  `keras.layers.Dropout`, `keras.activations.get`.
- `activation` continues to accept a string or a callable, resolved via
  `keras.activations.get` (concave = `-convex(-x)`; saturated synthesized as in
  the original).

**One deliberate internal deviation.** The original mutates `self.kernel` (swaps
the `Variable` for a tensor) inside `replace_kernel_using_monotonicity_indicator`
during `call`. Reassigning a `Variable` to a tensor is fragile across Keras 3
backends. The context-manager helper is retained as a public symbol for API
compatibility, but `MonoDense.call` computes
`ops.matmul(inputs, constrained_kernel) + bias` directly from its own weights —
numerically identical, no `Variable` mutation.

### Deprecation warning

- `DeprecationWarning` raised via `warnings.warn(..., stacklevel=2)` inside
  `MonoDense.__init__`, deduped with a module-level flag so it fires **once per
  process**.
- Message points users at the new layers (`mononet.torch`/`jax`/`keras`) and
  notes the `{-1, 0, 1}` → `±1` mask domain change.
- The warning fires on layer construction (and therefore transitively through
  `create_type_1/2`), **not** on import.

### Verification (committed goldens)

- `tools/gen-legacy-goldens.py`: a one-time, **manually-run** generator (not run
  in CI). It imports the original TF `monotonic-nn` and emits
  `(input, params) → output` reference vectors to `tests/legacy/goldens/`
  (`.npz` or JSON). A short header documents the exact source commit/version of
  the original used to generate them.
- `tests/legacy/test_equivalence.py`: loads the goldens, runs the port, asserts
  `allclose` within a fixed tolerance. Coverage:
  - all three indicator values, including `0`;
  - the three activation-split regimes (`is_convex`, `is_concave`, weighted
    3-split with default and custom `activation_weights`);
  - bias and no-bias;
  - `create_type_1` and `create_type_2` end-to-end.
  No `tensorflow` in CI.
- Standard unit tests: `pytest.warns(DeprecationWarning)` on construction;
  lazy-import assertion (`import mononet` does not import `keras`); `get_config`
  round-trip.

### Follow-up (release-gated)

Open a tracking issue in `davorrunje/mononet`: on public release, open a PR to
`airtai/monotonic-nn` replacing its implementation with `mononet` (making
`mononet` a dependency of the original), with `mononet.legacy` as the migration
bridge. This is recorded as a GitHub issue at the end of this brainstorm.

## Testing strategy

| Test | Anchors |
|---|---|
| `tests/legacy/test_equivalence.py` | Committed goldens from the original TF impl |
| `tests/legacy/test_warning.py` | `DeprecationWarning` fires once, on construction |
| `tests/legacy/test_lazy_import.py` | `import mononet` does not import `keras` or `mononet.legacy` |
| `tests/legacy/test_serialization.py` | `get_config` / `from_config` round-trip |

## Open questions

None outstanding — all resolved during brainstorming.
