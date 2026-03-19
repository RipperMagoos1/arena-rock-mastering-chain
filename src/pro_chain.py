import numpy as np


def clamp(audio, low=-1.0, high=1.0):
    return np.clip(audio, low, high)


def gentle_low_shelf(audio, boost=1.15):
    """
    Adds weight to lows without boom
    """
    out = np.copy(audio)
    out *= boost
    return clamp(out)


def soft_high_cut(audio, alpha=0.04):
    """
    Tames harsh highs
    """
    out = np.copy(audio)
    for ch in range(audio.shape[1]):
        for i in range(1, len(audio)):
            out[i, ch] = alpha * audio[i, ch] + (1 - alpha) * out[i - 1, ch]
    return clamp(out)


def bus_compressor(audio, threshold=0.28, ratio=2.5, makeup=1.1):
    """
    More natural compression (less squashed)
    """
    out = np.copy(audio)
    mono = np.mean(np.abs(audio), axis=1)

    env = np.zeros_like(mono)
    for i in range(1, len(mono)):
        env[i] = 0.1 * mono[i] + 0.9 * env[i - 1]

    gain = np.ones_like(env)
    for i in range(len(env)):
        if env[i] > threshold:
            over = env[i] - threshold
            compressed = threshold + over / ratio
            gain[i] = compressed / env[i]

    out[:, 0] *= gain
    out[:, 1] *= gain
    out *= makeup

    return clamp(out)


def parallel_compression(audio, amount=0.22):
    crushed = bus_compressor(audio, threshold=0.15, ratio=6.0, makeup=1.4)
    return clamp((1 - amount) * audio + amount * crushed)


def saturation(audio, drive=1.4, mix=0.25):
    wet = np.tanh(audio * drive)
    return clamp((1 - mix) * audio + mix * wet)


def stereo_widen(audio, width=1.08):
    left = audio[:, 0]
    right = audio[:, 1]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    side *= width

    new_left = mid + side
    new_right = mid - side

    return clamp(np.stack([new_left, new_right], axis=1))


def limiter(audio, ceiling=0.96):
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return clamp(audio)


def pro_master(audio):
    # 1. Keep original tone first
    audio = gentle_low_shelf(audio, boost=1.08)

    # 2. Light compression (glue, not squash)
    audio = bus_compressor(audio)

    # 3. Add body
    audio = parallel_compression(audio)

    # 4. Subtle saturation (not harsh)
    audio = saturation(audio)

    # 5. Slight width (not exaggerated)
    audio = stereo_widen(audio)

    # 6. Control highs LAST
    audio = soft_high_cut(audio)

    # 7. Final limiter
    audio = limiter(audio)

    return clamp(audio)