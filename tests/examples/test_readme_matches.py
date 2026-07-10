from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_readme_torch_block_matches_example_module() -> None:
    """README's torch example must equal docs/examples/risk_net_torch.py (sans SPDX)."""
    module_src = (_ROOT / "docs/examples/risk_net_torch.py").read_text()
    body = (
        module_src.split("\n", 1)[1] if module_src.startswith("# SPDX") else module_src
    )
    readme = (_ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", readme, re.DOTALL)
    assert any(block.strip() == body.strip() for block in blocks), (
        "README torch example drifted from docs/examples/risk_net_torch.py"
    )
