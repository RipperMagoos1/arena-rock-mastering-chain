from __future__ import annotations

import numpy as np
from .utils import db_to_lin, lin_to_db, clamp


def compressor(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -18.0,
    ratio: float = 3.0,
    attack_ms: float = 10.0,
    release_ms: float = 120.0,
    makeup_db: float = 0.0,
) -> np.ndarray:
    """
    Simple feed-forward RMS-ish compressor with attack/release smoothing.
    Works on stereo arrays shaped (n_samples, 2).
    """
    x = audio.astype(np.float32)
    thr = threshold_db
    r = max(1.0, float(ratio))

    attack = max(1.0, float(attack_ms)) / 1000.0
    release = max(1.0, float(release_ms)) / 1000.0

    a_a = np.exp(-1.0 / (sr * attack))
    a_r = np.exp(-1.0 / (sr * release))

    # detector: average channels then abs
    det = np.mean(np.abs(x), axis=1)
    det = np.maximum(det, 1e-8)

    env = 0.0
    gain = np.ones_like(det, dtype=np.float32)

    for i in range(det.shape[0]):
        d = det[i]
        # envelope follower
        if d > env:
            env = a_a * env + (1 - a_a) * d
        else:
            env = a_r * env + (1 - a_r) * d

        env_db = lin_to_db(env)
        over_db = env_db - thr
        if over_db > 0:
            gr_db = over_db - (over_db / r)  # gain reduction
            gain[i] = db_to_lin(-gr_db)
        else:
            gain[i] = 1.0

    g = gain[:, None]
    y = x * g
    y *= db_to_lin(makeup_db)
    return clamp(y)