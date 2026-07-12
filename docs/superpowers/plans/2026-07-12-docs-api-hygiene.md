# API-reference Cross-ref Hygiene + Core-object Docstrings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add field-list docstrings to the core value objects, drive the ~350 nitpicky cross-ref warnings to zero (resolve stable public types, regex-ignore irreducible external namespaces), and enforce it via `nitpicky=True` on the existing `-W` docs gate.

**Architecture:** Two independent halves — source docstrings (`mononet/core/{config,types}.py`) and build config (`docs/conf.py`). Docstrings land first so their annotation-derived cross-refs are accounted for when the config is driven to zero warnings and enforcement is switched on. Gates: `./tools/build-docs.sh` (strict `-W`), `uv run pytest tests/core`.

**Tech Stack:** Sphinx + `sphinx.ext.intersphinx` + `sphinx-autodoc2` (MyST render), MyST field-list docstrings, pytest.

**Spec:** [docs/superpowers/specs/2026-07-12-docs-api-hygiene-design.md](../specs/2026-07-12-docs-api-hygiene-design.md)

## Global Constraints

- **Branch:** `spec/docs-api-hygiene` (already checked out). Never commit to `main`.
- **Commit signing is broken in this container** — always `git commit --no-gpg-sign`.
- **No behavior change** to `mononet/**` — docstrings only. `tests/core` must stay green.
- **No CI-workflow change** — enforcement rides the existing `docs-smoke` job purely via `conf.py`'s `nitpicky=True`.
- **MyST field-list docstring format:** `:param <name>:`, `:returns:`, `:raises X:`; types come from signature annotations, never `:type:`/`:rtype:`; body text is MyST markdown; match the file's existing single-backtick inline-code style.
- **Ignore policy:** resolve stable public types via inventories; ignore irreducible external symbols with **namespace-scoped** `nitpick_ignore_regex` (not exact targets), each carrying a one-line justification comment.
- **Sequencing:** `nitpicky=True` is the LAST change, flipped only after `uv run sphinx-build -n` reports zero warnings.
- Run everything via `uv run`; never run any `uv sync` variant (prunes backend extras). Never `git add docs/_build/`.

---

### Task 1: Field-list docstrings on core value objects (#10)

**Files:**
- Modify: `mononet/core/config.py` (`MonoConfig`, `MonoResidualConfig` class docstrings)
- Modify: `mononet/core/types.py` (`MonotonicityMask`, `ActivationSpec`, `InitSpec`, and the `shape`/`__len__` methods)

**Interfaces:**
- Consumes: nothing. Produces: no API change — docstring text only.

- [ ] **Step 1: Confirm the baseline is green**

Run: `MONONET_TEST_BACKEND=torch uv run pytest tests/core -q`
Expected: PASS (the docstring edits must not change this).

- [ ] **Step 2: Replace the `MonoConfig` class docstring in `mononet/core/config.py`**

The current docstring is just `"""Hyperparameters for a single monotonic dense layer."""`. Replace it with:

```python
    """Hyperparameters for a single monotonic dense layer.

    :param units: Number of output units; must be positive.
    :param mode: Construction mode — `"absolute"` (the paper's `|W|`
        construction, default) or `"switch"` (the activation-switch variant).
    :param activation: Base activation applied by the layer (default
        `identity`).
    :param convex_fraction: Fraction of output units with a convex activation
        (absolute mode); must be in `[0, 1]`.
    :param init: Weight-initialization spec.
    :param bias: Whether the layer includes a bias term.
    :raises ValueError: If `units` is not positive, `mode` is unknown, or
        `convex_fraction` is outside `[0, 1]`.
    """
```

- [ ] **Step 3: Complete the `MonoResidualConfig` class docstring in `mononet/core/config.py`**

Replace the existing `MonoResidualConfig` docstring (which currently documents only `activation`) with:

```python
    """Hyperparameters for a dual-gated monotonic residual block.

    Gate fields are string tokens only; a custom callable gate or `F`
    module is not serialized.

    :param units: Number of output units; must be positive.
    :param mode: Construction mode for the default `F` — `"absolute"`
        (default) or `"switch"`.
    :param activation: Base activation for the default `F`. Required
        (keyword-only, no default) since a custom `F` is not representable
        here.
    :param alpha_gate: Gate token for the skip path.
    :param beta_gate: Gate token for the residual (transform) path.
    :param init: Weight-initialization spec.
    :raises ValueError: If `units` is not positive or `mode` is unknown.
    """
```

- [ ] **Step 4: Add field-lists to the `types.py` value objects**

In `mononet/core/types.py`, replace the `MonotonicityMask` class docstring with:

```python
    """Per-input-feature monotonicity specification.

    :param values: 1-D array of per-feature signs, each `+1` (output
        non-decreasing in this input) or `-1` (output non-increasing).
        Coerced to `int8`.
    :raises ValueError: If `values` is not 1-D or contains a value outside
        `{-1, +1}`.
    """
```

Replace the `ActivationSpec` class docstring with:

```python
    """Backend-agnostic activation specification.

    Backends resolve `name` to their own activation function.

    :param name: Activation name — one of `relu`, `elu`, `selu`, `softplus`,
        `identity`.
    :raises ValueError: If `name` is not a known activation.
    """
```

Replace the `InitSpec` class docstring with:

```python
    """Weight initialization specification.

    Backends resolve `scheme` to their own initializer.

    :param scheme: Initializer scheme — `he_normal` (default),
        `glorot_uniform`, or `lecun_normal`.
    :param seed: Optional RNG seed for reproducible initialization.
    """
```

Add a `:returns:` line to the `shape` property docstring (currently `"""Shape of the underlying mask array."""`):

```python
        """Shape of the underlying mask array.

        :returns: The shape tuple of the `int8` mask array.
        """
```

And to `__len__` (currently `"""Return the number of input features covered by this mask."""`):

```python
        """Return the number of input features covered by this mask.

        :returns: The number of input features (length of the mask).
        """
```

- [ ] **Step 5: Verify tests still pass and the docs still build**

```bash
MONONET_TEST_BACKEND=torch uv run pytest tests/core -q
./tools/build-docs.sh
```
Expected: `tests/core` PASS; `build succeeded`. (`nitpicky` is still off at this point, so the build passes even with unresolved xrefs — that is fixed in Task 2.)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format mononet/core/config.py mononet/core/types.py
uv run ruff check --exit-non-zero-on-fix
git add mononet/core/config.py mononet/core/types.py
git commit --no-gpg-sign -m "docs(core): field-list docstrings on config/types value objects"
```

---

### Task 2: Cross-ref hygiene + linkcheck_ignore, enforced via nitpicky (#5 + #11)

**Files:**
- Modify: `docs/conf.py`

**Interfaces:**
- Consumes: the Task 1 docstrings (their annotation-derived xrefs are part of the set driven to zero).

- [ ] **Step 1: Capture the baseline nitpick tally**

```bash
uv run sphinx-build -n -b html docs docs/_build/nitpick 2>&1 \
  | grep -E "WARNING.*reference target not found" \
  | sed -E 's/.*not found: //' | awk '{print $1}' | sort | uniq -c | sort -rn
```
Expected: a list dominated by `torch`, `typing_extensions`, `flax`, `numpy`, `jax`, `keras` targets (≈350 total). Keep this output as the working set.

- [ ] **Step 2: Add the flax inventory and drop the stale keras entry**

In `docs/conf.py`, replace the `intersphinx_mapping` block with:

```python
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "flax": ("https://flax.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    # keras.io is not a Sphinx site (no objects.inv); keras.* refs are
    # nitpick-ignored below rather than resolved.
}
intersphinx_disabled_reftypes = ["std:doc"]
```

Verify the flax inventory is actually reachable before relying on it:
```bash
curl -sI https://flax.readthedocs.io/en/latest/objects.inv | head -1
```
Expected: `HTTP/... 200`. If it is not 200, find flax's real docs base (its Read the Docs URL) and use that; if flax publishes no `objects.inv` at all, drop the flax mapping and instead ignore `flax\..*` by regex in Step 3 (note the decision in the report).

- [ ] **Step 3: Add the starting `nitpick_ignore_regex` and `linkcheck_ignore`**

Append to `docs/conf.py` (after the intersphinx block):

```python
# -- nitpicky cross-reference checking -------------------------------------
# Enabled at the end of this change (see below) once the build is warning-free.
nitpick_ignore_regex = [
    # No intersphinx inventory exists for these external namespaces, so their
    # cross-refs cannot be resolved. Ignore by NAMESPACE (not exact target) so a
    # dependency version bump cannot turn the strict -W docs gate red.
    ("py:.*", r"typing_extensions\..*"),  # no published objects.inv
    ("py:.*", r"keras\..*"),              # keras.io is not a Sphinx site
    ("py:.*", r"numpy\.typing\..*"),      # numpy.typing internals not in inventory
]

# -- linkcheck -------------------------------------------------------------
linkcheck_ignore = [
    # Bot-blocked (HTTP 403) but valid in a browser:
    r"https://patents\.justia\.com/patent/11551063",
    # pytorch docs use JS-generated anchors linkcheck cannot verify:
    r"https://docs\.pytorch\.org/docs/stable/.*#torch\..*",
]
```

- [ ] **Step 4: Iterate the ignore set to zero nitpick warnings**

Re-run the Step 1 command. For each remaining namespace in the output, decide:
- **Resolvable** (a stable public type in an inventory we map — torch/jax/numpy/flax): leave it; it should now resolve. If a *specific* public target still misses, it usually means the inventory lacks it (an internal path) → treat as the next case.
- **Irreducible** (framework-internal module paths, private paths, or a namespace with no inventory): add a tight namespace/prefix regex to `nitpick_ignore_regex`, with a one-line comment naming why it can't resolve.

Repeat until:
```bash
uv run sphinx-build -n -b html docs docs/_build/nitpick 2>&1 | grep -c "reference target not found"
```
prints `0`. Do **not** use a blanket `("py:.*", r".*")` — keep each regex scoped to a namespace/prefix that genuinely has no stable public target.

- [ ] **Step 5: Enable enforcement**

Add this line to `docs/conf.py` (in the nitpicky section):

```python
nitpicky = True
```

- [ ] **Step 6: Confirm the strict gate is green and linkcheck is clean**

```bash
./tools/build-docs.sh
```
Expected: `build succeeded` with **zero** warnings (nitpicky is now enforced by `-W`).

```bash
./tools/check-docs.sh
```
Expected: the nitpicky section reports no warnings; the linkcheck section no longer lists the justia URL or the pytorch `#torch.*` anchor URLs as broken (other pre-existing external redirects may remain).

- [ ] **Step 7: Spot-check a resolved link renders**

```bash
grep -o 'href="https://docs.pytorch.org[^"]*"' docs/_build/html/apidocs/mononet/mononet.torch.layers.html | head -1
```
Expected: a non-empty `href` to the pytorch docs — confirms `torch.Tensor`/`torch.nn.*` public types resolve to working links rather than being ignored.

- [ ] **Step 8: Commit**

```bash
git add docs/conf.py
git commit --no-gpg-sign -m "docs(build): resolve/ignore cross-refs, add linkcheck_ignore, enforce nitpicky"
```

---

## After all tasks

Do NOT open the PR from within a task — the finishing step
(superpowers:finishing-a-development-branch, after the whole-branch review)
handles push/PR. The PR should note that the audit follow-up list can tick off
#5, #10, and #11, and that `nitpicky=True` now means new docs referencing an
unmapped external type must add an inventory or a scoped ignore.

## Notes for the implementer

- `docs/_build/` is a build artifact — never `git add` it.
- The `nitpick_ignore_regex` entries are `(type_regex, target_regex)` tuples; `("py:.*", ...)` covers `py:class`/`py:obj`/`py:func` in one entry.
- If an intersphinx inventory fetch fails in the sandbox network, note it — but the existing torch/jax/numpy inventories already fetch in CI's `docs-smoke`, so flax (same kind of Read-the-Docs `objects.inv`) should fetch there too.
- Keep the resolve-vs-ignore balance honest: the goal is that the *important* public types (torch/jax/flax/numpy) stay clickable; only genuinely unmappable symbols get ignored.
