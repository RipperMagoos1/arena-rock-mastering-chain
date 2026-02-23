from __future__ import annotations

import numpy as np
from .utils import clamp


def soft_clip_saturation(audio: np.ndarray, drive: float = 0.35) -> np.ndarray:
    """
    Smooth saturation using tanh curve.
    drive: 0..1ish. Higher = more harmonic density.
    """
    d = float(max(0.0, drive))
    x = audio.astype(np.float32)

    # Drive gain, saturate, then normalize back a bit
    pre = 1.0 + (d * 6.0)
    y = np.tanh(x * pre)

    # Blend wet/dry for control
    mix = min(1.0, d + 0.15)
    out = (1 - mix) * x + mix * y
    return clamp(out)