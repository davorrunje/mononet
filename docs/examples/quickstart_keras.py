# SPDX-License-Identifier: Apache-2.0
"""Quickstart: a monotone regressor in Keras 3.

Non-decreasing in every one of its 4 inputs. Runs on whichever backend Keras is
configured to use (``KERAS_BACKEND``); ``MonoDense`` infers the input width.
"""

from __future__ import annotations

import keras

from mononet.keras import MonoDense, MonoResidual

model = keras.Sequential(
    [
        MonoDense(32, activation="elu"),
        MonoResidual(32, activation="elu"),
        MonoDense(1),
    ]
)

y = model(keras.ops.zeros((8, 4)))
print(tuple(y.shape))  # (8, 1) — monotone in all 4 inputs
