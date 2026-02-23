from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt
from .utils import db_to_lin


def _sos_highpass(sr: int, cutoff_hz: float, order: int = 2):
    cutoff = max(10.0, float(cutoff_hz))
    sos = butter(order, cutoff / (sr / 2.0), btype="highpass", output="sos")
    return sos


def _sos_lowpass(sr: int, cutoff_hz: float, order: int = 2):
    cutoff = min(float(cutoff_hz), sr / 2.2)
    sos = butter(order, cutoff / (sr / 2.0), btype="lowpass", output="sos")
    return sos


def apply_highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    sos = _sos_highpass(sr, cutoff_hz, order=2)
    return sosfilt(sos, audio, axis=0)


def _tilt_band(audio: np.ndarray, sr: int, split_hz: float, low_db: float, high_db: float) -> np.ndarray:
    """
    Simple "shelving-ish" tone shaping using band split and gain.
    This is not a true analog shelf; it's a stable approximation for a starter repo.
    """
    split = float(split_hz)
    low = sosfilt(_sos_lowpass(sr, split, order=2), audio, axis=0)
    high = audio - low
    low *= db_to_lin(low_db)
    high *= db_to_lin(high_db)
    return low + high


def apply_tone_curve(audio: np.ndarray, sr: int, low_shelf_hz: float, low_shelf_db: float,
                     presence_db: float, air_db: float) -> np.ndarray:
    """
    Broad-stroke EQ:
      - low shelf-ish at low_shelf_hz
      - presence-ish bump (implemented as mid/high tilt)
      - air-ish bump (implemented as high tilt)
    """
    # Low shelf approximation: split at low_shelf_hz and reduce low portion
    audio = _tilt_band(audio, sr, low_shelf_hz, low_shelf_db, 0.0)

    # Presence: boost above ~3-5k by tilting
    audio = _tilt_band(audio, sr, 3500.0, 0.0, presence_db)

    # Air: boost above ~10-12k by tilting again
    audio = _tilt_band(audio, sr, 11000.0, 0.0, air_db)

    return audio


def apply_mud_cut(audio: np.ndarray, sr: int, mud_hz: float, mud_cut_db: float) -> np.ndarray:
    """
    Gentle mud cut: reduce everything below mud_hz slightly (approx).
    """
    return _tilt_band(audio, sr, float(mud_hz), mud_cut_db, 0.0)