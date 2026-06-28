"""Deterministic per-backend seeding."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(backend: str, seed: int) -> None:
    """Seed Python, NumPy, and the active backend's RNG.

    :param backend: Backend name ("torch", "keras", or "jax").
    :param seed: Seed value to use.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if backend == "torch":
        import torch

        torch.manual_seed(seed)
    elif backend == "keras":
        import keras

        keras.utils.set_random_seed(seed)
    # jax is functional: callers thread an explicit key from `seed`.
