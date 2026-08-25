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
