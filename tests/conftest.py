# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures for mononet tests."""

import os

# Prevent OpenMP library conflicts when both PyTorch and XGBoost are loaded
# in the same process (macOS ARM). Harmless on Linux CI.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")


# Add project-wide fixtures below.
# Example:
#
# @pytest.fixture
# def sample_data() -> dict:
#     return {"key": "value"}
