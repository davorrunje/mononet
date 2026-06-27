# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the top-level package import surface."""

from __future__ import annotations


def test_import_mononet_succeeds_without_any_backend() -> None:
    """`import mononet` must succeed even with no backend installed.

    The package must not eagerly import torch/jax/keras at module load.
    """
    import mononet

    assert isinstance(mononet.__version__, str)
    assert mononet.__version__ != ""


def test_no_backend_modules_imported_at_top_level() -> None:
    """Verify torch/jax/keras are not pulled in by `import mononet`."""
    import sys

    # Drop any previously imported mononet sub-modules so the test is
    # deterministic regardless of test order.
    for name in list(sys.modules):
        if name == "mononet" or name.startswith("mononet."):
            del sys.modules[name]

    import mononet  # noqa: F401

    assert "mononet.torch" not in sys.modules
    assert "mononet.jax" not in sys.modules
    assert "mononet.keras" not in sys.modules


def test_public_re_exports_core_symbols() -> None:
    """Importing core symbols from `mononet` (top-level) must work."""
    from mononet import MonotonicityMask  # noqa: F401
