"""Deterministic seed handling."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set common Python, NumPy, and Torch seeds when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        return
