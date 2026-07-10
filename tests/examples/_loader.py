"""Load a docs/examples/*.py module by file path (kept out of the package)."""

from __future__ import annotations

import importlib.util
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType

_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "docs" / "examples"


def load_example(filename: str) -> ModuleType:
    """Import a ``docs/examples`` module from its file path.

    :param filename: File name under ``docs/examples`` (e.g. ``risk_net_torch.py``).
    :returns: The imported module object.
    """
    path = _EXAMPLES / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
