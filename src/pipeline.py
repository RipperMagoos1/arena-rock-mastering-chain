from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf

from .utils import ensure_stereo, load_preset, clamp
from .eq import apply_highpass, apply_tone_curve, apply_mud_cut
from .compression import compressor
from .saturation import soft_clip_saturation
from .stereo import widen
from .limiter import limiter


def arena_master(input_wav: str, output_wav: str, preset: str = "broadcast_master") -> None:
    """
    Load WAV -> apply EQ/comp/sat/stereo/limiter -> write WAV
    preset can be:
      - "broadcast_master"  (loads presets/broadcast_master.json)
      - "arena_mix"         (loads presets/arena_mix.json)
      - or a path to a JSON preset file
    """
    in_path = Path(input_wav)
    out_path = Path(output_wav)

    if preset in ("broadcast_master", "arena_mix"):
        preset_path = Path("presets") / f"{preset}.json"
    else:
        preset_path = Path(preset)

    p = load_preset(preset_path)

    audio, sr = sf.read(str(in_path), always_2d=False)
    audio = ensure_stereo(np.asarray(audio, dtype=np.float32))

    # --- EQ ---
    audio = apply_highpass(audio, sr, p.get("hp_cut_hz", 30))
    audio = apply_mud_cut(audio, sr, p.get("mud_cut_hz", 280), p.get("mud_cut_db", -2.5))
    audio = apply_tone_curve(
        audio, sr,
        p.get("low_shelf_hz", 90),
        p.get("low_shelf_db", -1.5),
        p.get("presence_db", 1.5),
        p.get("air_db", 1.2),
    )

    # --- Compression ---
    audio = compressor(
        audio, sr,
        threshold_db=p.get("compressor_threshold_db", -18),
        ratio=p.get("compressor_ratio", 3.0),
        attack_ms=p.get("compressor_attack_ms", 10),
        release_ms=p.get("compressor_release_ms", 120),
        makeup_db=p.get("compressor_makeup_db", 2.0),
    )

    # --- Saturation ---
    audio = soft_clip_saturation(audio, drive=p.get("saturation_drive", 0.35))

    # --- Stereo ---
    audio = widen(audio, width=p.get("stereo_width", 0.10))

    # --- Limiter ---
    audio = limiter(
        audio, sr,
        ceiling_db=p.get("limiter_ceiling_db", -1.0),
        release_ms=p.get("limiter_release_ms", 80),
    )

    audio = clamp(audio)
    sf.write(str(out_path), audio, sr)