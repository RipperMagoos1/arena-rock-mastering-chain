from __future__ import annotations

import numpy as np
from .utils import clamp


def widen(audio: np.ndarray, width: float = 0.10) -> np.ndarray:
    """
    Mid/Side widening while staying relatively mono-safe.
    width ~ 0.0 to 0.2 is a sane range for masters.
    """
    w = float(np.clip(width, 0.0, 0.5))
    x = audio.astype(np.float32)

    L = x[:, 0]
    R = x[:, 1]
    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)

    side *= (1.0 + (w * 2.0))

    L2 = mid + side
    R2 = mid - side
    y = np.stack([L2, R2], axis=1)
    return clamp(y)