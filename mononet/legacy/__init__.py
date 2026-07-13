# SPDX-License-Identifier: Apache-2.0
"""Legacy compatibility layer for the original ``airtai/monotonic-nn`` API.

Importing this module pulls in Keras 3. It is intentionally NOT imported by
``import mononet`` — the top-level package stays backend-free. Every symbol
here is deprecated; use the modern ``mononet`` backends instead.
"""

from mononet.legacy.mono_dense_layer import MonoDense

__all__ = ["MonoDense"]
