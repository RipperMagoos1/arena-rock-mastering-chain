from src.wav_io import read_wav, write_wav
from src.saturation import soft_clip_saturation
from src.stereo import widen
from src.compression import compressor
from src.utils import clamp
import numpy as np


def lowpass_simple(audio, alpha=0.02):
    out = np.copy(audio)
    for ch in range(audio.shape[1]):
        for i in range(1, len(audio)):
            out[i, ch] = alpha * audio[i, ch] + (1 - alpha) * out[i - 1, ch]
    return out


def add_body(audio, amount=0.10):
    """
    Adds low-mid fullness without making it muddy.
    """
    body = lowpass_simple(audio, alpha=0.03) - lowpass_simple(audio, alpha=0.008)
    return clamp(audio + amount * body)


def add_low_support(audio, amount=0.05):
    """
    Adds a small amount of low-end support.
    """
    lows = lowpass_simple(audio, alpha=0.006)
    return clamp(audio + amount * lows)


def simple_eq(audio):
    """
    Keep the original feel, just slightly fuller.
    """
    audio = audio * 0.98
    audio = add_body(audio, amount=0.10)
    audio = add_low_support(audio, amount=0.04)
    return clamp(audio)


def simple_limiter(audio):
    peak = np.max(np.abs(audio))
    if peak > 0.96:
        audio = audio * (0.96 / peak)
    return clamp(audio)


if __name__ == "__main__":
    audio, sr = read_wav("input.wav")

    # Original chain + slight fullness upgrade
    audio = simple_eq(audio)
    audio = compressor(
        audio,
        sr,
        threshold_db=-18.0,
        ratio=3.0,
        attack_ms=10.0,
        release_ms=120.0,
        makeup_db=1.2,
    )
    audio = soft_clip_saturation(audio, 0.32)
    audio = widen(audio, 0.08)
    audio = simple_limiter(audio)

    write_wav("output_master.wav", audio, sr)

    print("Done: output_master.wav created")