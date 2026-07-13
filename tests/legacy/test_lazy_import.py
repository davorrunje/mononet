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
