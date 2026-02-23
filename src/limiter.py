from __future__ import annotations

import numpy as np
from .utils import db_to_lin, clamp


def limiter(audio: np.ndarray, sr: int, ceiling_db: float = -1.0, release_ms: float = 80.0) -> np.ndarray:
    """
    Simple look-ahead-less limiter (fast peak clamp + release smoothing).
    Not a true true-peak limiter, but a clean starter for the repo.
    """
    x = audio.astype(np.float32)
    ceiling = db_to_lin(float(ceiling_db))
    rel = max(5.0, float(release_ms)) / 1000.0
    a_r = np.exp(-1.0 / (sr * rel))

    g = 1.0
    y = np.zeros_like(x)

    for i in range(x.shape[0]):
        peak = float(np.max(np.abs(x[i, :])))
        if peak > 1e-8:
            target_g = min(1.0, ceiling / peak)
        else:
            target_g = 1.0

        # instantaneous attack, smoothed release
        if target_g < g:
            g = target_g
        else:
            g = a_r * g + (1 - a_r) * target_g

        y[i, :] = x[i, :] * g

    return clamp(y, -ceiling, ceiling)