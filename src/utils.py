from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np


def db_to_lin(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def lin_to_db(x: float, floor_db: float = -120.0) -> float:
    x = max(float(x), 1e-12)
    db = 20.0 * np.log10(x)
    return float(max(db, floor_db))


def clamp(x: np.ndarray, lo: float = -1.0, hi: float = 1.0) -> np.ndarray:
    return np.clip(x, lo, hi)


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """
    Returns audio shaped (n_samples, 2).
    Accepts (n_samples,) mono or (n_samples, channels).
    """
    if audio.ndim == 1:
        return np.stack([audio, audio], axis=-1)
    if audio.ndim == 2 and audio.shape[1] == 1:
        return np.concatenate([audio, audio], axis=1)
    if audio.ndim == 2 and audio.shape[1] >= 2:
        return audio[:, :2]
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def load_preset(preset_path: str | Path) -> dict:
    p = Path(preset_path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)