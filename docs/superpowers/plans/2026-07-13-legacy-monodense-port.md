# Legacy MonoDense Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `mononet.legacy` — a faithful, backend-agnostic (Keras 3 `keras.ops`) reproduction of the original `airtai/monotonic-nn` `MonoDense` layer, its network builders, and helper functions, emitting a `DeprecationWarning` on use and verified numerically identical to the original via committed goldens.

**Architecture:** A single implementation module `mononet/legacy/mono_dense_layer.py` (mirrors the original module name) reproduces the original public surface using `keras.ops`. `MonoDense` subclasses `keras.layers.Dense` for weight-name/config fidelity but overrides `call` to compute `ops.matmul(inputs, constrained_kernel) + bias` directly (no `Variable` mutation). `mononet/legacy/__init__.py` re-exports the public API. Nothing is added to the top-level `mononet/__init__.py`, preserving lazy backend imports. Numeric equivalence is anchored by committed goldens generated once from the original TF implementation.

**Tech Stack:** Python 3.11+, Keras 3 (`keras.ops`, backend-agnostic), pytest, uv. Golden generation uses `monotonic-nn==0.3.5` + `tensorflow` in an ephemeral `uv run --with` environment (never added to the project).

## Global Constraints

- Python 3.11+, line length 88 (ruff). Strict mypy; type hints on every function/method.
- **MyST field-list docstrings** on all public functions/classes: `:param x:`, `:returns:`, `:raises X:`. No `:type:`/`:rtype:`. Body is MyST markdown.
- SPDX header `# SPDX-License-Identifier: Apache-2.0` as the first line of every new `.py` source file (see existing files). Test files start with the same SPDX header.
- **Do not reintroduce Pydantic.** Stdlib only for value objects.
- **Preserve lazy backend imports:** `import mononet` must NOT import `keras` or `mononet.legacy`. Never add legacy/backend imports to `mononet/__init__.py`.
- **Do not use `MonotonicityMask`** in legacy code — the legacy layer keeps the raw `{-1, 0, 1}` `monotonicity_indicator` semantics.
- Keras tests set `os.environ.setdefault("KERAS_BACKEND", "jax")` before importing keras, then `pytest.importorskip("keras")`. Follow this pattern in every legacy test.
- Numeric tolerance for golden comparison: `np.allclose(got, expected, atol=1e-5, rtol=1e-5)` on float32.
- Commit with `git commit --no-gpg-sign` (devcontainer signing is unavailable). Never commit to `main` — all work on branch `legacy-monodense-port` (already checked out).
- The original source of truth for behavior is `airt/_components/mono_dense_layer.py` at `monotonic-nn==0.3.5` (repo `airtai/monotonic-nn`). Its public `__all__`: `get_saturated_activation`, `get_activation_functions`, `apply_activations`, `get_monotonicity_indicator`, `apply_monotonicity_indicator_to_kernel`, `replace_kernel_using_monotonicity_indicator`, `MonoDense` — plus builder classmethods `MonoDense.create_type_1` / `MonoDense.create_type_2`.

---

## File Structure

- `mononet/legacy/__init__.py` — public re-exports; no heavy imports at module top beyond the impl module (which imports keras lazily-per-posture: importing `mononet.legacy` is what pulls keras, and that is acceptable — only `import mononet` must stay clean).
- `mononet/legacy/mono_dense_layer.py` — the full port: helpers, `MonoDense`, builders.
- `tools/gen-legacy-goldens.py` — one-time, manually-run golden generator (uses the original TF impl).
- `tests/legacy/__init__.py` — SPDX header only.
- `tests/legacy/goldens/monodense_cases.json` — committed goldens for `MonoDense`.
- `tests/legacy/goldens/builder_cases.json` — committed goldens for `create_type_1/2`.
- `tests/legacy/test_helpers.py` — unit tests for the module-level helper functions.
- `tests/legacy/test_mono_dense.py` — `MonoDense` behavior + deprecation warning + config round-trip.
- `tests/legacy/test_builders.py` — `create_type_1/2` structure + forward pass.
- `tests/legacy/test_equivalence.py` — golden comparison (MonoDense + builders).
- `tests/legacy/test_lazy_import.py` — `import mononet` does not import keras / mononet.legacy.

---

## Task 1: Legacy package scaffold + lazy-import guarantee

**Files:**
- Create: `mononet/legacy/__init__.py`
- Create: `mononet/legacy/mono_dense_layer.py`
- Create: `tests/legacy/__init__.py`
- Create: `tests/legacy/test_lazy_import.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `mononet.legacy` importable; `mono_dense_layer` module exists with a `_warn_once() -> None` stub used by later tasks.

- [ ] **Step 1: Write the failing lazy-import test**

`tests/legacy/test_lazy_import.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""`import mononet` must not eagerly import keras or the legacy module."""

from __future__ import annotations

import sys


def test_import_mononet_does_not_import_legacy_or_keras() -> None:
    for name in list(sys.modules):
        if name == "mononet" or name.startswith("mononet.") or name == "keras":
            del sys.modules[name]

    import mononet  # noqa: F401

    assert "mononet.legacy" not in sys.modules
    assert "keras" not in sys.modules


def test_importing_legacy_exposes_public_api() -> None:
    from mononet.legacy import MonoDense  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/legacy/test_lazy_import.py -v`
Expected: FAIL on `test_importing_legacy_exposes_public_api` with `ModuleNotFoundError: No module named 'mononet.legacy'`.

- [ ] **Step 3: Create the scaffold**

`tests/legacy/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
```

`mononet/legacy/mono_dense_layer.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Legacy MonoDense layer — a faithful, backend-agnostic reproduction of the
original ``airtai/monotonic-nn`` implementation.

This module reproduces the original public API (``MonoDense`` plus builders and
helpers) using ``keras.ops`` so it runs under any Keras 3 backend. It exists
solely as a migration bridge; new code should use ``mononet.torch`` /
``mononet.jax`` / ``mononet.keras`` instead. Every ``MonoDense`` construction
emits a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

_WARNED = False

_DEPRECATION_MESSAGE = (
    "mononet.legacy.MonoDense reproduces the original airtai/monotonic-nn layer "
    "for migration only. Prefer mononet.torch/jax/keras (MonoLinear/MonoDense). "
    "Note the monotonicity spec changed: the legacy {-1, 0, 1} indicator maps to "
    "a two-value +/-1 mask in the new layers."
)


def _warn_once() -> None:
    """Emit the legacy :class:`DeprecationWarning` at most once per process."""
    global _WARNED
    if not _WARNED:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)
        _WARNED = True
```

`mononet/legacy/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Legacy compatibility layer for the original ``airtai/monotonic-nn`` API.

Importing this module pulls in Keras 3. It is intentionally NOT imported by
``import mononet`` — the top-level package stays backend-free. Every symbol
here is deprecated; use the modern ``mononet`` backends instead.
"""

from mononet.legacy.mono_dense_layer import MonoDense

__all__ = ["MonoDense"]
```

Add a temporary minimal `MonoDense` placeholder to `mono_dense_layer.py` so the import resolves (it will be replaced in Task 4):

```python
class MonoDense:  # placeholder, replaced in Task 4
    """Placeholder replaced by the real layer in a later task."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `MONONET_TEST_BACKEND=keras uv run pytest tests/legacy/test_lazy_import.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add mononet/legacy tests/legacy/__init__.py tests/legacy/test_lazy_import.py
git commit --no-gpg-sign -m "feat(legacy): scaffold mononet.legacy package with lazy-import guarantee"
```

---

## Task 2: Activation helpers (`get_saturated_activation`, `get_activation_functions`, `apply_activations`)

**Files:**
- Modify: `mononet/legacy/mono_dense_layer.py`
- Test: `tests/legacy/test_helpers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `get_saturated_activation(convex_activation, concave_activation, a: float = 1.0, c: float = 1.0) -> Callable`
  - `get_activation_functions(activation: str | Callable | None = None) -> tuple[Callable, Callable, Callable]` returning `(convex, concave, saturated)`
  - `apply_activations(x, *, units: int, convex_activation, concave_activation, saturated_activation, is_convex: bool = False, is_concave: bool = False, activation_weights: tuple[float, float, float] = (7.0, 7.0, 2.0))`

- [ ] **Step 1: Write the failing tests**

`tests/legacy/test_helpers.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for legacy helper functions."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

from mononet.legacy.mono_dense_layer import (  # noqa: E402
    apply_activations,
    get_activation_functions,
    get_saturated_activation,
)


def test_get_activation_functions_relu_convex_concave() -> None:
    convex, concave, _ = get_activation_functions("relu")
    x = ops.convert_to_tensor(np.array([-2.0, -1.0, 1.0, 2.0], dtype="float32"))
    # convex = relu; concave(x) = -relu(-x)
    assert np.allclose(np.asarray(convex(x)), [0.0, 0.0, 1.0, 2.0])
    assert np.allclose(np.asarray(concave(x)), [-2.0, -1.0, 0.0, 0.0])


def test_saturated_activation_is_continuous_at_zero() -> None:
    convex, concave, saturated = get_activation_functions("elu")
    x = ops.convert_to_tensor(np.array([-1e-6, 0.0, 1e-6], dtype="float32"))
    y = np.asarray(saturated(x))
    assert abs(y[0] - y[2]) < 1e-3  # continuous across the x<=0 boundary


def test_apply_activations_convex_uses_all_convex_split() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.convert_to_tensor(np.array([[-1.0, -2.0, 3.0]], dtype="float32"))
    y = np.asarray(
        apply_activations(
            x,
            units=3,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            is_convex=True,
        )
    )
    assert np.allclose(y, [[0.0, 0.0, 3.0]])  # all-convex == relu


def test_apply_activations_weighted_split_sizes() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.convert_to_tensor(np.arange(10, dtype="float32").reshape(1, 10))
    # weights (7,7,2)/16 * 10 -> round(4.375)=4, round(4.375)=4, remainder=2
    y = np.asarray(
        apply_activations(
            x,
            units=10,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            activation_weights=(7.0, 7.0, 2.0),
        )
    )
    assert y.shape == (1, 10)


def test_apply_activations_rejects_bad_weights() -> None:
    convex, concave, saturated = get_activation_functions("relu")
    x = ops.zeros((1, 4))
    with pytest.raises(ValueError):
        apply_activations(
            x,
            units=4,
            convex_activation=convex,
            concave_activation=concave,
            saturated_activation=saturated,
            activation_weights=(1.0, -1.0, 1.0),
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/legacy/test_helpers.py -v`
Expected: FAIL with `ImportError: cannot import name 'apply_activations'`.

- [ ] **Step 3: Implement the activation helpers**

Add to `mononet/legacy/mono_dense_layer.py` (after the `_warn_once` block; add imports at top):

```python
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from keras import activations, ops

if TYPE_CHECKING:
    from collections.abc import Generator
```

```python
def get_saturated_activation(
    convex_activation: Callable[[Any], Any],
    concave_activation: Callable[[Any], Any],
    a: float = 1.0,
    c: float = 1.0,
) -> Callable[[Any], Any]:
    """Build the saturated activation from a convex/concave pair.

    :param convex_activation: Convex, monotonically increasing base activation.
    :param concave_activation: Its concave reflection ``-f(-x)``.
    :param a: Output scale.
    :param c: Knot location of the piecewise join.
    :returns: A callable mapping a tensor to the saturated activation.
    """

    def saturated_activation(x: Any) -> Any:
        cc = convex_activation(ops.ones_like(x) * c)
        return a * ops.where(
            x <= 0,
            convex_activation(x + c) - cc,
            concave_activation(x - c) + cc,
        )

    return saturated_activation


@lru_cache
def get_activation_functions(
    activation: str | Callable[[Any], Any] | None = None,
) -> tuple[Callable[[Any], Any], Callable[[Any], Any], Callable[[Any], Any]]:
    """Resolve the convex, concave, and saturated activations for a base name.

    :param activation: Base activation name or callable (assumed convex,
        monotonically increasing). ``None`` resolves to the linear activation.
    :returns: ``(convex_activation, concave_activation, saturated_activation)``.
    """
    convex_activation = activations.get(
        activation.lower() if isinstance(activation, str) else activation
    )

    def concave_activation(x: Any) -> Any:
        return -convex_activation(-x)

    saturated_activation = get_saturated_activation(
        convex_activation, concave_activation
    )
    return convex_activation, concave_activation, saturated_activation


def apply_activations(
    x: Any,
    *,
    units: int,
    convex_activation: Callable[[Any], Any],
    concave_activation: Callable[[Any], Any],
    saturated_activation: Callable[[Any], Any],
    is_convex: bool = False,
    is_concave: bool = False,
    activation_weights: tuple[float, float, float] = (7.0, 7.0, 2.0),
) -> Any:
    """Split ``x`` into convex/concave/saturated groups and activate each.

    :param x: Pre-activation tensor of shape ``(batch, units)``.
    :param units: Output width (the size of the last axis of ``x``).
    :param convex_activation: Convex branch activation.
    :param concave_activation: Concave branch activation.
    :param saturated_activation: Saturated branch activation.
    :param is_convex: Force an all-convex split ``(units, 0, 0)``.
    :param is_concave: Force an all-concave split ``(0, units, 0)``.
    :param activation_weights: Relative sizes of the three groups; ignored when
        ``is_convex`` or ``is_concave`` is set.
    :returns: Activated tensor of shape ``(batch, units)``.
    :raises ValueError: If ``activation_weights`` is not length 3 or has a
        negative entry.
    """
    if convex_activation is None:
        return x

    if is_convex:
        normalized_activation_weights = np.array([1.0, 0.0, 0.0])
    elif is_concave:
        normalized_activation_weights = np.array([0.0, 1.0, 0.0])
    else:
        if len(activation_weights) != 3:
            raise ValueError(f"activation_weights={activation_weights}")
        if (np.array(activation_weights) < 0).any():
            raise ValueError(f"activation_weights={activation_weights}")
        normalized_activation_weights = np.array(activation_weights) / sum(
            activation_weights
        )

    s_convex = round(normalized_activation_weights[0] * units)
    s_concave = round(normalized_activation_weights[1] * units)
    s_saturated = units - s_convex - s_concave

    # keras.ops.split takes cut points (numpy semantics), not sizes.
    x_convex, x_concave, x_saturated = ops.split(
        x, [s_convex, s_convex + s_concave], axis=-1
    )

    y_convex = convex_activation(x_convex)
    y_concave = concave_activation(x_concave)
    y_saturated = saturated_activation(x_saturated)

    return ops.concatenate([y_convex, y_concave, y_saturated], axis=-1)
```

Note: the original computed an unused `ccc` inside `saturated_activation`; it is intentionally dropped (dead code, no behavioral effect).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/legacy/test_helpers.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Run lint + type check**

Run: `uv run ruff check mononet/legacy tests/legacy && uv run ruff format mononet/legacy tests/legacy && uv run mypy mononet/legacy`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add mononet/legacy/mono_dense_layer.py tests/legacy/test_helpers.py
git commit --no-gpg-sign -m "feat(legacy): port activation helpers (saturated/convex/concave split)"
```

---

## Task 3: Monotonicity-indicator helpers

**Files:**
- Modify: `mononet/legacy/mono_dense_layer.py`
- Test: `tests/legacy/test_helpers.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `get_monotonicity_indicator(monotonicity_indicator, *, input_shape, units) -> np.ndarray`
  - `apply_monotonicity_indicator_to_kernel(kernel, monotonicity_indicator)` → constrained kernel tensor
  - `replace_kernel_using_monotonicity_indicator(layer, monotonicity_indicator)` → context manager

- [ ] **Step 1: Write the failing tests**

Append to `tests/legacy/test_helpers.py`:

```python
from mononet.legacy.mono_dense_layer import (  # noqa: E402
    apply_monotonicity_indicator_to_kernel,
    get_monotonicity_indicator,
)


def test_get_monotonicity_indicator_reshapes_to_column() -> None:
    ind = get_monotonicity_indicator([1, -1, 0], input_shape=(None, 3), units=4)
    assert ind.shape == (3, 1)


def test_get_monotonicity_indicator_rejects_out_of_domain() -> None:
    with pytest.raises(ValueError):
        get_monotonicity_indicator([2], input_shape=(None, 1), units=1)


def test_get_monotonicity_indicator_rejects_rank_gt_2() -> None:
    with pytest.raises(ValueError):
        get_monotonicity_indicator(
            np.ones((2, 2, 2)), input_shape=(None, 2), units=2
        )


def test_apply_indicator_to_kernel_signs() -> None:
    kernel = ops.convert_to_tensor(
        np.array([[-1.0, 2.0], [3.0, -4.0]], dtype="float32")
    )
    indicator = ops.convert_to_tensor(np.array([[1], [-1]], dtype="float32"))
    out = np.asarray(apply_monotonicity_indicator_to_kernel(kernel, indicator))
    # row 0 -> |.| (increasing); row 1 -> -|.| (decreasing)
    assert np.allclose(out, [[1.0, 2.0], [-3.0, -4.0]])


def test_apply_indicator_zero_leaves_kernel_unchanged() -> None:
    kernel = ops.convert_to_tensor(np.array([[-1.0, 2.0]], dtype="float32"))
    indicator = ops.convert_to_tensor(np.array([[0]], dtype="float32"))
    out = np.asarray(apply_monotonicity_indicator_to_kernel(kernel, indicator))
    assert np.allclose(out, [[-1.0, 2.0]])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/legacy/test_helpers.py -k indicator -v`
Expected: FAIL with `ImportError: cannot import name 'get_monotonicity_indicator'`.

- [ ] **Step 3: Implement the indicator helpers**

Add to `mononet/legacy/mono_dense_layer.py`:

```python
from contextlib import contextmanager  # add to the import block at top
```

```python
def get_monotonicity_indicator(
    monotonicity_indicator: Any,
    *,
    input_shape: tuple[int, ...],
    units: int,
) -> np.ndarray:
    """Normalise a monotonicity indicator to a broadcastable column vector.

    :param monotonicity_indicator: Scalar or array of per-input signs, each in
        ``{-1, 0, 1}`` (``1`` increasing, ``-1`` decreasing, ``0`` free).
    :param input_shape: Layer input shape; ``input_shape[-1]`` is the fan-in.
    :param units: Output width.
    :returns: The indicator reshaped to ``(fan_in, 1)`` (or as given if already
        2-D), validated against ``{-1, 0, 1}``.
    :raises ValueError: If the indicator has rank > 2 or contains a value
        outside ``{-1, 0, 1}``.
    """
    monotonicity_indicator = np.array(monotonicity_indicator)
    if len(monotonicity_indicator.shape) < 2:
        monotonicity_indicator = np.reshape(monotonicity_indicator, (-1, 1))
    elif len(monotonicity_indicator.shape) > 2:
        raise ValueError(
            "monotonicity_indicator has rank greater than 2: "
            f"{monotonicity_indicator.shape}"
        )

    np.broadcast_to(monotonicity_indicator, shape=(input_shape[-1], units))

    if not np.all(
        (monotonicity_indicator == -1)
        | (monotonicity_indicator == 0)
        | (monotonicity_indicator == 1)
    ):
        raise ValueError(
            "Each element of monotonicity_indicator must be one of -1, 0, 1, "
            f"but it is: '{monotonicity_indicator}'"
        )
    return monotonicity_indicator


def apply_monotonicity_indicator_to_kernel(
    kernel: Any,
    monotonicity_indicator: Any,
) -> Any:
    """Sign-constrain a kernel by a monotonicity indicator.

    :param kernel: Weight tensor of shape ``(fan_in, units)``.
    :param monotonicity_indicator: Broadcastable ``{-1, 0, 1}`` indicator.
    :returns: Kernel with ``|W|`` where the indicator is ``1``, ``-|W|`` where
        it is ``-1``, and ``W`` unchanged where it is ``0``.
    """
    monotonicity_indicator = ops.convert_to_tensor(monotonicity_indicator)
    abs_kernel = ops.abs(kernel)
    xs = ops.where(monotonicity_indicator == 1, abs_kernel, kernel)
    xs = ops.where(monotonicity_indicator == -1, -abs_kernel, xs)
    return xs


@contextmanager
def replace_kernel_using_monotonicity_indicator(
    layer: Any,
    monotonicity_indicator: Any,
) -> Generator[None, None, None]:
    """Temporarily swap ``layer.kernel`` for its sign-constrained version.

    Retained for API compatibility with the original package. The ported
    :class:`MonoDense` does not rely on this context manager (it constrains the
    kernel functionally in ``call`` instead).

    :param layer: A layer exposing a mutable ``kernel`` attribute.
    :param monotonicity_indicator: Broadcastable ``{-1, 0, 1}`` indicator.
    :yields: Nothing; restores the original kernel on exit.
    """
    old_kernel = layer.kernel
    layer.kernel = apply_monotonicity_indicator_to_kernel(
        layer.kernel, monotonicity_indicator
    )
    try:
        yield
    finally:
        layer.kernel = old_kernel
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/legacy/test_helpers.py -v`
Expected: PASS (all helper tests).

- [ ] **Step 5: Lint + type check + commit**

```bash
uv run ruff check mononet/legacy tests/legacy && uv run ruff format mononet/legacy tests/legacy && uv run mypy mononet/legacy
git add mononet/legacy/mono_dense_layer.py tests/legacy/test_helpers.py
git commit --no-gpg-sign -m "feat(legacy): port monotonicity-indicator helpers"
```

---

## Task 4: `MonoDense` layer + deprecation warning + config round-trip

**Files:**
- Modify: `mononet/legacy/mono_dense_layer.py` (replace the Task 1 placeholder `MonoDense`)
- Test: `tests/legacy/test_mono_dense.py`

**Interfaces:**
- Consumes: `apply_activations`, `get_activation_functions`, `get_monotonicity_indicator`, `apply_monotonicity_indicator_to_kernel`, `_warn_once`.
- Produces: `class MonoDense(keras.layers.Dense)` with constructor
  `MonoDense(units, *, activation=None, monotonicity_indicator=1, is_convex=False, is_concave=False, activation_weights=(7.0, 7.0, 2.0), **kwargs)` and `get_config()` returning keys `units, activation, monotonicity_indicator, is_convex, is_concave, activation_weights`.

- [ ] **Step 1: Write the failing tests**

`tests/legacy/test_mono_dense.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Behavior, deprecation warning, and serialization for legacy MonoDense."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")
from keras import ops  # noqa: E402

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense  # noqa: E402


def test_construction_emits_deprecation_warning() -> None:
    legacy._WARNED = False
    with pytest.warns(DeprecationWarning, match="mononet.legacy"):
        MonoDense(4)


def test_warning_fires_only_once() -> None:
    legacy._WARNED = False
    with pytest.warns(DeprecationWarning):
        MonoDense(4)
    with warnings_none():
        MonoDense(4)  # second construction: no warning


import contextlib  # noqa: E402
import warnings  # noqa: E402


@contextlib.contextmanager
def warnings_none():  # type: ignore[no-untyped-def]
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        yield


def test_rejects_convex_and_concave() -> None:
    legacy._WARNED = True
    with pytest.raises(ValueError):
        MonoDense(4, is_convex=True, is_concave=True)


def test_rejects_bad_activation_weights() -> None:
    legacy._WARNED = True
    with pytest.raises(ValueError):
        MonoDense(4, activation_weights=(1.0, 1.0))  # not length 3
    with pytest.raises(ValueError):
        MonoDense(4, activation_weights=(-1.0, 1.0, 1.0))


def test_forward_pass_is_increasing_for_positive_indicator() -> None:
    legacy._WARNED = True
    layer = MonoDense(1, activation="relu", monotonicity_indicator=1)
    layer.build((None, 3))
    x0 = ops.convert_to_tensor(np.zeros((1, 3), dtype="float32"))
    x1 = ops.convert_to_tensor(np.ones((1, 3), dtype="float32"))
    y0 = float(np.asarray(layer(x0))[0, 0])
    y1 = float(np.asarray(layer(x1))[0, 0])
    assert y1 >= y0  # non-decreasing along an all-increasing input


def test_get_config_round_trip() -> None:
    legacy._WARNED = True
    layer = MonoDense(
        5,
        activation="elu",
        monotonicity_indicator=1,
        is_convex=True,
        activation_weights=(2.0, 3.0, 1.0),
    )
    cfg = layer.get_config()
    assert cfg["units"] == 5
    assert cfg["activation"] == "elu"
    assert cfg["is_convex"] is True
    assert cfg["activation_weights"] == (2.0, 3.0, 1.0)
    rebuilt = MonoDense.from_config(cfg)
    assert rebuilt.units == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/legacy/test_mono_dense.py -v`
Expected: FAIL (placeholder `MonoDense` takes no args / is not a keras layer).

- [ ] **Step 3: Replace the placeholder with the real `MonoDense`**

In `mono_dense_layer.py`, add `import keras` to the top import block and replace the placeholder class:

```python
class MonoDense(keras.layers.Dense):  # type: ignore[misc]
    """Monotonic counterpart of ``keras.layers.Dense`` (legacy API).

    Faithful reproduction of the original ``airtai/monotonic-nn`` layer. The
    kernel is sign-constrained per :paramref:`monotonicity_indicator` and the
    output is split into convex/concave/saturated activation groups.

    :param units: Output dimensionality.
    :param activation: Base activation (assumed convex, monotonically
        increasing), e.g. ``"relu"`` or ``"elu"``; name or callable. ``None``
        is linear.
    :param monotonicity_indicator: Per-input sign in ``{-1, 0, 1}`` (``1``
        increasing, ``-1`` decreasing, ``0`` non-monotonic). Scalar or array.
    :param is_convex: Force an all-convex activation split.
    :param is_concave: Force an all-concave activation split.
    :param activation_weights: Relative sizes of the convex/concave/saturated
        groups; ignored when ``is_convex`` or ``is_concave`` is set.
    :raises ValueError: If both ``is_convex`` and ``is_concave`` are set, or if
        ``activation_weights`` is not length 3 or has a negative entry.
    """

    def __init__(
        self,
        units: int,
        *,
        activation: str | Callable[[Any], Any] | None = None,
        monotonicity_indicator: Any = 1,
        is_convex: bool = False,
        is_concave: bool = False,
        activation_weights: tuple[float, float, float] = (7.0, 7.0, 2.0),
        **kwargs: Any,
    ) -> None:
        """Construct a legacy MonoDense layer (emits a DeprecationWarning)."""
        _warn_once()
        if is_convex and is_concave:
            raise ValueError(
                "The model cannot be set to be both convex and concave "
                "(only linear functions are both)."
            )
        if len(activation_weights) != 3:
            raise ValueError(
                "There must be exactly three components of activation_weights, "
                f"but we have this instead: {activation_weights}."
            )
        if (np.array(activation_weights) < 0).any():
            raise ValueError(
                "Values of activation_weights must be non-negative, but we have "
                f"this instead: {activation_weights}."
            )

        super().__init__(units=units, activation=None, **kwargs)

        self.units = units
        self.org_activation = activation
        self.monotonicity_indicator = monotonicity_indicator
        self.is_convex = is_convex
        self.is_concave = is_concave
        self.activation_weights = activation_weights

        (
            self.convex_activation,
            self.concave_activation,
            self.saturated_activation,
        ) = get_activation_functions(self.org_activation)

    def build(self, input_shape: Any) -> None:
        """Create the Dense weights and normalise the indicator.

        :param input_shape: Shape tuple; ``input_shape[-1]`` is the fan-in.
        """
        super().build(input_shape)
        self.monotonicity_indicator = get_monotonicity_indicator(
            monotonicity_indicator=self.monotonicity_indicator,
            input_shape=input_shape,
            units=self.units,
        )

    def call(self, inputs: Any) -> Any:
        """Apply the sign-constrained affine map and grouped activations.

        :param inputs: Input tensor of shape ``(batch, ..., fan_in)``.
        :returns: Output tensor of shape ``(batch, ..., units)``.
        """
        constrained_kernel = apply_monotonicity_indicator_to_kernel(
            self.kernel, self.monotonicity_indicator
        )
        h = ops.matmul(inputs, constrained_kernel)
        if self.use_bias:
            h = h + self.bias
        return apply_activations(
            h,
            units=self.units,
            convex_activation=self.convex_activation,
            concave_activation=self.concave_activation,
            saturated_activation=self.saturated_activation,
            is_convex=self.is_convex,
            is_concave=self.is_concave,
            activation_weights=self.activation_weights,
        )

    def get_config(self) -> dict[str, Any]:
        """Serialize the layer configuration.

        :returns: Config dict with the original legacy keys.
        """
        return {
            "units": self.units,
            "activation": self.org_activation,
            "monotonicity_indicator": self.monotonicity_indicator,
            "is_convex": self.is_convex,
            "is_concave": self.is_concave,
            "activation_weights": self.activation_weights,
        }
```

Remove the temporary placeholder class from Task 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/legacy/test_mono_dense.py -v`
Expected: PASS (all).

- [ ] **Step 5: Lint + type check + commit**

```bash
uv run ruff check mononet/legacy tests/legacy && uv run ruff format mononet/legacy tests/legacy && uv run mypy mononet/legacy
git add mononet/legacy/mono_dense_layer.py tests/legacy/test_mono_dense.py
git commit --no-gpg-sign -m "feat(legacy): port MonoDense layer with deprecation warning"
```

---

## Task 5: Network builders `create_type_1` / `create_type_2`

**Files:**
- Modify: `mononet/legacy/mono_dense_layer.py`
- Modify: `mononet/legacy/__init__.py` (export builders + helpers)
- Test: `tests/legacy/test_builders.py`

**Interfaces:**
- Consumes: `MonoDense`.
- Produces:
  - `create_type_1(inputs, *, units, final_units, activation, n_layers, final_activation=None, monotonicity_indicator=1, is_convex=False, is_concave=False, dropout=None)` → output tensor
  - `create_type_2(inputs, *, input_units=None, units, final_units, activation, n_layers, final_activation=None, monotonicity_indicator=1, is_convex=False, is_concave=False, dropout=None)` → output tensor
  - `MonoDense.create_type_1` / `MonoDense.create_type_2` classmethods delegating to the module functions.
  - internal `_create_mono_block`, `_prepare_mono_input_n_param`, `_check_convexity_params`.

- [ ] **Step 1: Write the failing tests**

`tests/legacy/test_builders.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Structure and forward-pass tests for legacy network builders."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense, create_type_1, create_type_2  # noqa: E402


# NOTE: the original API's per-feature `monotonicity_indicator` list is only
# supported with a list/dict of per-feature single-column Input tensors (its
# actual upstream usage) — NOT a single multi-feature tensor. The builders are
# a faithful port, so tests must use that same calling convention. A single
# tensor is only valid with a scalar indicator (int), which the classmethod
# test below exercises.
def test_create_type_1_builds_runnable_model() -> None:
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(4)]
    out = create_type_1(
        inputs,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=3,
        monotonicity_indicator=[1, 1, -1, 0],
    )
    model = keras.Model(inputs, out)
    xs = [np.zeros((2, 1), dtype="float32") for _ in range(4)]
    assert tuple(model(xs).shape) == (2, 1)


def test_create_type_1_accepts_dict_inputs() -> None:
    legacy._WARNED = True
    inputs = {"a": keras.Input(shape=(1,)), "b": keras.Input(shape=(1,))}
    out = create_type_1(
        inputs,
        units=4,
        final_units=1,
        activation="elu",
        n_layers=2,
        monotonicity_indicator={"a": 1, "b": -1},
    )
    model = keras.Model(inputs, out)
    y = model(
        {
            "a": np.zeros((2, 1), dtype="float32"),
            "b": np.zeros((2, 1), dtype="float32"),
        }
    )
    assert tuple(y.shape) == (2, 1)


def test_create_type_1_classmethod_matches_function() -> None:
    legacy._WARNED = True
    # single tensor + scalar (int) indicator is the one single-tensor case the
    # original supports; default monotonicity_indicator=1.
    inp = keras.Input(shape=(3,))
    out = MonoDense.create_type_1(
        inp, units=4, final_units=2, activation="relu", n_layers=2
    )
    model = keras.Model(inp, out)
    assert tuple(model(np.zeros((1, 3), dtype="float32")).shape) == (1, 2)


def test_create_type_2_builds_runnable_model() -> None:
    legacy._WARNED = True
    inputs = [keras.Input(shape=(1,)) for _ in range(4)]
    out = create_type_2(
        inputs,
        units=8,
        final_units=1,
        activation="elu",
        n_layers=2,
        monotonicity_indicator=[1, -1, 0, 1],
    )
    model = keras.Model(inputs, out)
    xs = [np.zeros((2, 1), dtype="float32") for _ in range(4)]
    assert tuple(model(xs).shape) == (2, 1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/legacy/test_builders.py -v`
Expected: FAIL with `ImportError: cannot import name 'create_type_1'`.

- [ ] **Step 3: Implement the builders**

Add to `mono_dense_layer.py` (imports: add `TypeVar` to the typing import; add `from keras.layers import Concatenate, Dense, Dropout`):

```python
T = TypeVar("T")


def _create_mono_block(
    *,
    units: list[int],
    activation: str | Callable[[Any], Any],
    monotonicity_indicator: Any = 1,
    is_convex: bool = False,
    is_concave: bool = False,
    dropout: float | None = None,
) -> Callable[[Any], Any]:
    """Return a callable that stacks ``MonoDense`` layers over an input tensor.

    :param units: Output width of each layer, in order.
    :param activation: Base activation for the hidden layers (the final layer is
        linear).
    :param monotonicity_indicator: Indicator for the first layer; ``1`` for all
        subsequent layers (monotonicity is preserved thereafter).
    :param is_convex: Force convex activation split.
    :param is_concave: Force concave activation split.
    :param dropout: Optional dropout rate applied between hidden layers.
    :returns: A function mapping an input tensor to the block output.
    """

    def create_mono_block_inner(x: Any) -> Any:
        if len(units) == 0:
            return x
        y = x
        for i in range(len(units)):
            y = MonoDense(
                units=units[i],
                activation=activation if i < len(units) - 1 else None,
                monotonicity_indicator=monotonicity_indicator if i == 0 else 1,
                is_convex=is_convex,
                is_concave=is_concave,
                name=f"mono_dense_{i}"
                + ("_increasing" if i != 0 else "")
                + ("_convex" if is_convex else "")
                + ("_concave" if is_concave else ""),
            )(y)
            if (i < len(units) - 1) and dropout:
                y = Dropout(dropout)(y)
        return y

    return create_mono_block_inner


def _prepare_mono_input_n_param(
    inputs: Any,
    param: Any,
) -> tuple[list[Any], list[Any], list[str]]:
    """Broadcast a per-input parameter against a list/dict/tensor of inputs.

    :param inputs: A tensor, a list of tensors, or a dict of named tensors.
    :param param: A scalar, list, or dict matching ``inputs``.
    :returns: ``(inputs_list, param_list, sorted_feature_names)``.
    :raises ValueError: On incompatible input/param types or mismatched lengths.
    """
    if isinstance(inputs, list):
        if isinstance(param, int):
            param = [param] * len(inputs)
        elif isinstance(param, list):
            if len(inputs) != len(param):
                raise ValueError(f"{len(inputs)} != {len(param)}")
        else:
            raise ValueError(
                f"Incompatible types: {type(inputs)=}, {type(param)=}"
            )
        sorted_feature_names = [f"{i}" for i in range(len(inputs))]
    elif isinstance(inputs, dict):
        sorted_feature_names = sorted(inputs.keys())
        if isinstance(param, int):
            param = [param] * len(inputs)
        elif isinstance(param, dict):
            if set(param.keys()) != set(sorted_feature_names):
                raise ValueError(
                    f"{set(param.keys())} != {set(sorted_feature_names)}"
                )
            param = [param[k] for k in sorted_feature_names]
        else:
            raise ValueError(
                f"Incompatible types: {type(inputs)=}, {type(param)=}"
            )
        inputs = [inputs[k] for k in sorted_feature_names]
    else:
        if not isinstance(param, int):
            raise ValueError(
                f"Incompatible types: {type(inputs)=}, {type(param)=}"
            )
        inputs = [inputs]
        param = [param]
        sorted_feature_names = ["inputs"]
    return inputs, param, sorted_feature_names


def _check_convexity_params(
    monotonicity_indicator: list[int],
    is_convex: list[bool],
    is_concave: list[bool],
    names: list[str],
) -> tuple[bool, bool]:
    """Validate per-input convexity flags and reduce them to block-level flags.

    :param monotonicity_indicator: Per-input indicators.
    :param is_convex: Per-input convex flags.
    :param is_concave: Per-input concave flags.
    :param names: Per-input names (for error messages).
    :returns: ``(has_convex, has_concave)`` for the shared mono block.
    :raises ValueError: If any input is marked both convex and concave.
    """
    ix = [
        i
        for i in range(len(monotonicity_indicator))
        if is_convex[i] and is_concave[i]
    ]
    if len(ix) > 0:
        raise ValueError(
            f"Parameters both convex and concave: {[names[i] for i in ix]}"
        )
    has_convex = any(is_convex)
    has_concave = any(is_concave)
    if has_convex and has_concave:
        print("WARNING: we have both convex and concave parameters")
    return has_convex, has_concave


def create_type_1(
    inputs: Any,
    *,
    units: int,
    final_units: int,
    activation: str | Callable[[Any], Any],
    n_layers: int,
    final_activation: str | Callable[[Any], Any] | None = None,
    monotonicity_indicator: Any = 1,
    is_convex: Any = False,
    is_concave: Any = False,
    dropout: float | None = None,
) -> Any:
    """Build a Type-1 monotonic MLP (features concatenated, then mono block).

    :param inputs: Input tensor, list of tensors, or dict of named tensors.
    :param units: Hidden-layer width.
    :param final_units: Output-layer width.
    :param activation: Base activation.
    :param n_layers: Total layers (hidden + output).
    :param final_activation: Optional activation applied to the output.
    :param monotonicity_indicator: Per-input indicator (int, list, or dict).
    :param is_convex: Per-input convex flag(s).
    :param is_concave: Per-input concave flag(s).
    :param dropout: Optional dropout rate between hidden layers.
    :returns: Output tensor.
    """
    _, is_convex, _ = _prepare_mono_input_n_param(inputs, is_convex)
    _, is_concave, _ = _prepare_mono_input_n_param(inputs, is_concave)
    x, monotonicity_indicator, names = _prepare_mono_input_n_param(
        inputs, monotonicity_indicator
    )
    has_convex, has_concave = _check_convexity_params(
        monotonicity_indicator, is_convex, is_concave, names
    )
    y = Concatenate()(x)
    y = _create_mono_block(
        units=[units] * (n_layers - 1) + [final_units],
        activation=activation,
        monotonicity_indicator=monotonicity_indicator,
        is_convex=has_convex,
        is_concave=has_concave and not has_convex,
        dropout=dropout,
    )(y)
    if final_activation is not None:
        y = activations.get(final_activation)(y)
    return y


def create_type_2(
    inputs: Any,
    *,
    input_units: int | None = None,
    units: int,
    final_units: int,
    activation: str | Callable[[Any], Any],
    n_layers: int,
    final_activation: str | Callable[[Any], Any] | None = None,
    monotonicity_indicator: Any = 1,
    is_convex: Any = False,
    is_concave: Any = False,
    dropout: float | None = None,
) -> Any:
    """Build a Type-2 monotonic network (per-input units, then shared block).

    :param inputs: Input tensor, list of tensors, or dict of named tensors.
    :param input_units: Per-input preprocessing width (default ``max(units//4, 1)``).
    :param units: Hidden-layer width.
    :param final_units: Output-layer width.
    :param activation: Base activation.
    :param n_layers: Total layers (hidden + output) in the shared block.
    :param final_activation: Optional activation applied to the output.
    :param monotonicity_indicator: Per-input indicator (int, list, or dict).
    :param is_convex: Per-input convex flag(s).
    :param is_concave: Per-input concave flag(s).
    :param dropout: Optional dropout rate between hidden layers.
    :returns: Output tensor.
    """
    _, is_convex, _ = _prepare_mono_input_n_param(inputs, is_convex)
    _, is_concave, _ = _prepare_mono_input_n_param(inputs, is_concave)
    x, monotonicity_indicator, names = _prepare_mono_input_n_param(
        inputs, monotonicity_indicator
    )
    has_convex, has_concave = _check_convexity_params(
        monotonicity_indicator, is_convex, is_concave, names
    )
    if input_units is None:
        input_units = max(units // 4, 1)
    y = [
        (
            MonoDense(
                units=input_units,
                activation=activation,
                monotonicity_indicator=monotonicity_indicator[i],
                is_convex=is_convex[i],
                is_concave=is_concave[i],
                name=f"mono_dense_{names[i]}"
                + (
                    "_increasing"
                    if monotonicity_indicator[i] == 1
                    else "_decreasing"
                )
                + ("_convex" if is_convex[i] else "")
                + ("_concave" if is_concave[i] else ""),
            )(x[i])
            if monotonicity_indicator[i] != 0
            else Dense(
                units=input_units, activation=activation, name=f"dense_{names[i]}"
            )(x[i])
        )
        for i in range(len(x))
    ]
    y = Concatenate(name="preprocessed_features")(y)
    monotonicity_indicator_block: list[int] = sum(
        ([abs(v)] * input_units for v in monotonicity_indicator), []
    )
    y = _create_mono_block(
        units=[units] * (n_layers - 1) + [final_units],
        activation=activation,
        monotonicity_indicator=monotonicity_indicator_block,
        is_convex=has_convex,
        is_concave=has_concave and not has_convex,
        dropout=dropout,
    )(y)
    if final_activation is not None:
        y = activations.get(final_activation)(y)
    return y
```

Attach the classmethods (after the class body is defined — place at end of module):

```python
MonoDense.create_type_1 = classmethod(  # type: ignore[assignment]
    lambda cls, inputs, **kwargs: create_type_1(inputs, **kwargs)
)
MonoDense.create_type_2 = classmethod(  # type: ignore[assignment]
    lambda cls, inputs, **kwargs: create_type_2(inputs, **kwargs)
)
```

Update `mononet/legacy/__init__.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Legacy compatibility layer for the original ``airtai/monotonic-nn`` API.

Importing this module pulls in Keras 3. It is intentionally NOT imported by
``import mononet`` — the top-level package stays backend-free. Every symbol
here is deprecated; use the modern ``mononet`` backends instead.
"""

from mononet.legacy.mono_dense_layer import (
    MonoDense,
    apply_activations,
    apply_monotonicity_indicator_to_kernel,
    create_type_1,
    create_type_2,
    get_activation_functions,
    get_monotonicity_indicator,
    get_saturated_activation,
    replace_kernel_using_monotonicity_indicator,
)

__all__ = [
    "MonoDense",
    "apply_activations",
    "apply_monotonicity_indicator_to_kernel",
    "create_type_1",
    "create_type_2",
    "get_activation_functions",
    "get_monotonicity_indicator",
    "get_saturated_activation",
    "replace_kernel_using_monotonicity_indicator",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/legacy/test_builders.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Run the full legacy suite + lint + type check**

Run: `uv run pytest tests/legacy -v && uv run ruff check mononet/legacy tests/legacy && uv run ruff format mononet/legacy tests/legacy && uv run mypy mononet/legacy`
Expected: all pass, no lint/type errors.

- [ ] **Step 6: Commit**

```bash
git add mononet/legacy tests/legacy/test_builders.py
git commit --no-gpg-sign -m "feat(legacy): port create_type_1/create_type_2 network builders"
```

---

## Task 6: Golden generator + committed goldens + equivalence test

**Files:**
- Create: `tools/gen-legacy-goldens.py`
- Create: `tests/legacy/goldens/monodense_cases.json`
- Create: `tests/legacy/goldens/builder_cases.json`
- Test: `tests/legacy/test_equivalence.py`

**Interfaces:**
- Consumes: `MonoDense`, `create_type_1`, `create_type_2` (ported).
- Produces: committed golden JSON files and a test asserting the port matches them.

**Golden JSON schema (`monodense_cases.json`):** a list of objects, each:
```json
{
  "name": "relu_inc_bias",
  "units": 4,
  "activation": "relu",
  "monotonicity_indicator": [1, 1, -1, 0],
  "is_convex": false,
  "is_concave": false,
  "activation_weights": [7.0, 7.0, 2.0],
  "use_bias": true,
  "kernel": [[...], ...],      // shape (in_f, units)
  "bias": [...],               // shape (units,), omitted if use_bias false
  "input": [[...], ...],       // shape (batch, in_f)
  "output": [[...], ...]       // shape (batch, units)
}
```
**Golden JSON schema (`builder_cases.json`):** a list of objects, each:
```json
{
  "name": "type1_basic",
  "builder": "type_1",
  "kwargs": { "units": 8, "final_units": 1, "activation": "elu",
              "n_layers": 3, "monotonicity_indicator": [1, 1, -1, 0] },
  "n_features": 4,           // number of per-feature single-column inputs
  "weights": [[...], ...],   // flat list of get_weights() arrays, in order
  "input": [[...], ...],     // shape (batch, n_features); split into columns
  "output": [[...], ...]
}
```

- [ ] **Step 1: Write the golden generator**

`tools/gen-legacy-goldens.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""One-time generator for legacy MonoDense golden vectors.

Run manually in an ephemeral environment with the ORIGINAL TensorFlow package —
this is NOT run in CI:

    uv run --with 'monotonic-nn==0.3.5' --with tensorflow \\
        python tools/gen-legacy-goldens.py

It imports the original ``airt`` implementation, runs a fixed battery of cases,
and writes JSON goldens under ``tests/legacy/goldens/``. The ported layer is
asserted equal to these vectors in ``tests/legacy/test_equivalence.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from airt.keras.layers import MonoDense  # original TF implementation

GOLDENS = Path(__file__).resolve().parent.parent / "tests" / "legacy" / "goldens"


def _monodense_cases() -> list[dict]:
    rng = np.random.default_rng(0)
    specs = [
        dict(name="relu_inc_bias", units=4, activation="relu",
             monotonicity_indicator=[1, 1, -1, 0], use_bias=True,
             activation_weights=(7.0, 7.0, 2.0), in_f=4),
        dict(name="elu_convex", units=6, activation="elu",
             monotonicity_indicator=1, is_convex=True, use_bias=True,
             activation_weights=(7.0, 7.0, 2.0), in_f=3),
        dict(name="elu_concave_nobias", units=6, activation="elu",
             monotonicity_indicator=1, is_concave=True, use_bias=False,
             activation_weights=(7.0, 7.0, 2.0), in_f=3),
        dict(name="relu_custom_weights", units=10, activation="relu",
             monotonicity_indicator=[1, -1, 0], use_bias=True,
             activation_weights=(2.0, 3.0, 1.0), in_f=3),
    ]
    out = []
    for s in specs:
        layer = MonoDense(
            s["units"], activation=s["activation"],
            monotonicity_indicator=s["monotonicity_indicator"],
            is_convex=s.get("is_convex", False),
            is_concave=s.get("is_concave", False),
            activation_weights=s["activation_weights"],
            use_bias=s["use_bias"],
        )
        layer.build((None, s["in_f"]))
        kernel = rng.standard_normal((s["in_f"], s["units"])).astype("float32")
        weights = [kernel]
        if s["use_bias"]:
            bias = rng.standard_normal((s["units"],)).astype("float32")
            weights.append(bias)
        layer.set_weights(weights)
        x = rng.standard_normal((5, s["in_f"])).astype("float32")
        y = np.asarray(layer(tf.convert_to_tensor(x)))
        case = dict(
            name=s["name"], units=s["units"], activation=s["activation"],
            monotonicity_indicator=s["monotonicity_indicator"],
            is_convex=s.get("is_convex", False),
            is_concave=s.get("is_concave", False),
            activation_weights=list(s["activation_weights"]),
            use_bias=s["use_bias"],
            kernel=kernel.tolist(), input=x.tolist(), output=y.tolist(),
        )
        if s["use_bias"]:
            case["bias"] = weights[1].tolist()
        out.append(case)
    return out


def _builder_cases() -> list[dict]:
    # The original builders take a list/dict of per-feature single-column
    # Input tensors (their real usage) — NOT a single multi-feature tensor.
    from airt.keras.layers import MonoDense as M
    rng = np.random.default_rng(1)
    out = []
    specs = [
        dict(name="type1_basic", builder="type_1", n_features=4,
             kwargs=dict(units=8, final_units=1, activation="elu",
                         n_layers=3, monotonicity_indicator=[1, 1, -1, 0])),
        dict(name="type2_basic", builder="type_2", n_features=4,
             kwargs=dict(units=8, final_units=1, activation="elu",
                         n_layers=2, monotonicity_indicator=[1, -1, 0, 1])),
    ]
    for s in specs:
        n = s["n_features"]
        inputs = [tf.keras.Input(shape=(1,)) for _ in range(n)]
        build = M.create_type_1 if s["builder"] == "type_1" else M.create_type_2
        y = build(inputs, **s["kwargs"])
        model = tf.keras.Model(inputs, y)
        weights = [w.tolist() for w in model.get_weights()]
        x = rng.standard_normal((5, n)).astype("float32")
        feed = [tf.convert_to_tensor(x[:, i:i + 1]) for i in range(n)]
        out.append(dict(
            name=s["name"], builder=s["builder"], kwargs=s["kwargs"],
            n_features=n, weights=weights,
            input=x.tolist(),
            output=np.asarray(model(feed)).tolist(),
        ))
    return out


def main() -> None:
    GOLDENS.mkdir(parents=True, exist_ok=True)
    (GOLDENS / "monodense_cases.json").write_text(
        json.dumps(_monodense_cases(), indent=2)
    )
    (GOLDENS / "builder_cases.json").write_text(
        json.dumps(_builder_cases(), indent=2)
    )
    print(f"wrote goldens to {GOLDENS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the goldens (requires the original TF package)**

Run: `uv run --with 'monotonic-nn==0.3.5' --with tensorflow python tools/gen-legacy-goldens.py`
Expected: `wrote goldens to .../tests/legacy/goldens`, and two JSON files created.

> If `monotonic-nn==0.3.5` fails to resolve against the current Python, pin an interpreter it supports, e.g. prefix with `--python 3.11`. Record the exact command used in the commit message. This step is a maintainer step; the resulting JSON is committed so CI never needs TF.

- [ ] **Step 3: Write the equivalence test**

`tests/legacy/test_equivalence.py`:

```python
# SPDX-License-Identifier: Apache-2.0
"""Assert the ported layer matches committed goldens from the original impl."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("KERAS_BACKEND", "jax")
keras = pytest.importorskip("keras")

import mononet.legacy.mono_dense_layer as legacy  # noqa: E402
from mononet.legacy import MonoDense, create_type_1, create_type_2  # noqa: E402

GOLDENS = Path(__file__).parent / "goldens"


def _load(name: str) -> list[dict]:
    return json.loads((GOLDENS / name).read_text())


@pytest.mark.parametrize("case", _load("monodense_cases.json"),
                         ids=lambda c: c["name"])
def test_monodense_matches_golden(case: dict) -> None:
    legacy._WARNED = True
    layer = MonoDense(
        case["units"], activation=case["activation"],
        monotonicity_indicator=case["monotonicity_indicator"],
        is_convex=case["is_convex"], is_concave=case["is_concave"],
        activation_weights=tuple(case["activation_weights"]),
        use_bias=case["use_bias"],
    )
    x = np.array(case["input"], dtype="float32")
    layer.build((None, x.shape[-1]))
    weights = [np.array(case["kernel"], dtype="float32")]
    if case["use_bias"]:
        weights.append(np.array(case["bias"], dtype="float32"))
    layer.set_weights(weights)
    got = np.asarray(layer(keras.ops.convert_to_tensor(x)))
    assert np.allclose(got, np.array(case["output"]), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("case", _load("builder_cases.json"),
                         ids=lambda c: c["name"])
def test_builder_matches_golden(case: dict) -> None:
    legacy._WARNED = True
    n = case["n_features"]
    inputs = [keras.Input(shape=(1,)) for _ in range(n)]
    build = create_type_1 if case["builder"] == "type_1" else create_type_2
    out = build(inputs, **case["kwargs"])
    model = keras.Model(inputs, out)
    model.set_weights([np.array(w, dtype="float32") for w in case["weights"]])
    x = np.array(case["input"], dtype="float32")
    feed = [keras.ops.convert_to_tensor(x[:, i : i + 1]) for i in range(n)]
    got = np.asarray(model(feed))
    assert np.allclose(got, np.array(case["output"]), atol=1e-5, rtol=1e-5)
```

- [ ] **Step 4: Run the equivalence test**

Run: `uv run pytest tests/legacy/test_equivalence.py -v`
Expected: PASS (all parametrized cases). If a builder case fails on weight ordering, inspect `model.get_weights()` ordering parity between original and port; the shared `MonoDense`/`Dense` graph produces the same order — adjust only if a real mismatch is found.

- [ ] **Step 5: Run the full legacy suite + lint + type check**

Run: `uv run pytest tests/legacy -v && uv run ruff check mononet tests/legacy && uv run mypy mononet/legacy`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tools/gen-legacy-goldens.py tests/legacy/goldens tests/legacy/test_equivalence.py
git commit --no-gpg-sign -m "test(legacy): committed goldens + equivalence vs original TF impl"
```

---

## Task 7: Docs surface + final full-suite verification

**Files:**
- Modify: `docs/` API index if legacy needs an autodoc entry (check `docs/apidocs` generation config).
- No new code.

**Interfaces:** none.

- [ ] **Step 1: Check whether autodoc picks up `mononet.legacy` automatically**

Run: `./tools/build-docs.sh 2>&1 | tail -30`
Expected: docs build succeeds. If `mononet.legacy` is not analysed and it should be, add it to the autodoc2 package list in `docs/conf.py` (mirror how `mononet.keras` is configured).

- [ ] **Step 2: Run the full test suite across backends**

Run: `uv run pytest tests/legacy -v`
Then confirm nothing else broke: `uv run pytest -m "not slow"`
Expected: all pass; no import-time regressions in `tests/test_top_level_imports.py`.

- [ ] **Step 3: Run all pre-commit hooks**

Run: `uv run pre-commit run --all-files`
Expected: all hooks pass (note: the equivalence `REFERENCE_HASH` hook is unrelated to legacy and should remain green).

- [ ] **Step 4: Commit any docs changes**

```bash
git add docs
git commit --no-gpg-sign -m "docs(legacy): expose mononet.legacy in API reference"
```

- [ ] **Step 5: Open the PR**

Follow `PULL_REQUEST_GUIDE.md`. Title: `feat(legacy): mononet.legacy drop-in port of original MonoDense`. Body references the spec (`docs/superpowers/specs/2026-07-13-legacy-monodense-port-design.md`) and follow-up issue #102.

---

## Self-Review

**Spec coverage:**
- Placement/namespace + lazy import → Task 1. ✓
- Keras 3 port of MonoDense (subclass Dense, `call` override, no Variable mutation) → Task 4. ✓
- Helpers (all six) → Tasks 2–3. ✓
- Builders create_type_1/2 + classmethods → Task 5. ✓
- Raw `{-1,0,1}` indicator, no MonotonicityMask → Tasks 3–4. ✓
- DeprecationWarning once, on construction → Tasks 1 + 4. ✓
- Committed goldens + generator (no TF in CI) → Task 6. ✓
- Serialization round-trip, warning, lazy-import tests → Tasks 1, 4. ✓
- Exclude experiments/training → not in any task (correctly out of scope). ✓
- Follow-up issue → already created (#102). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases" — all steps contain concrete code and commands.

**Type consistency:** `_warn_once`, `get_activation_functions` (3-tuple), `apply_activations` (keyword-only), `get_monotonicity_indicator` (returns reshaped array), `apply_monotonicity_indicator_to_kernel`, `MonoDense(units, *, ...)`, `create_type_1/2` signatures are consistent across the tasks that define and consume them. Golden JSON schema in Task 6 matches both the generator (Step 1) and the consumer test (Step 3).
