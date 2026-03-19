import numpy as np


def clamp(audio, low=-1.0, high=1.0):
    return np.clip(audio, low, high)


def highpass_simple(audio, strength=0.995):
    """
    Very simple DC/rumble reduction.
    """
    out = np.copy(audio)
    out[1:] = audio[1:] - strength * audio[:-1]
    return out


def lowpass_simple(audio, alpha=0.08):
    """
    Very simple smoothing filter for harsh highs.
    Lower alpha = darker.
    """
    out = np.copy(audio)
    for ch in range(audio.shape[1]):
        for i in range(1, len(audio)):
            out[i, ch] = alpha * audio[i, ch] + (1 - alpha) * out[i - 1, ch]
    return out


def eq_stage(audio):
    """
    Simple tone shaping:
    - reduce sub rumble
    - tame harsh top
    - add presence by blending
    """
    rumble_cut = highpass_simple(audio, strength=0.995)
    smooth = lowpass_simple(rumble_cut, alpha=0.18)

    # Blend dry + smoothed for cleaner top
    shaped = 0.78 * rumble_cut + 0.22 * smooth

    # Small upper-mid/presence feel by parallel emphasis
    presence = shaped - lowpass_simple(shaped, alpha=0.04)
    shaped = shaped + 0.08 * presence

    return clamp(shaped)


def envelope(signal, attack_coeff=0.2, release_coeff=0.002):
    env = np.zeros_like(signal)
    for i in range(1, len(signal)):
        x = abs(signal[i])
        if x > env[i - 1]:
            env[i] = attack_coeff * x + (1 - attack_coeff) * env[i - 1]
        else:
            env[i] = release_coeff * x + (1 - release_coeff) * env[i - 1]
    return env


def bus_compressor(audio, threshold=0.22, ratio=3.0, makeup=1.25):
    """
    Stereo bus compression.
    """
    out = np.copy(audio)
    mono = np.mean(np.abs(audio), axis=1)
    env = envelope(mono, attack_coeff=0.2, release_coeff=0.002)

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


def parallel_compression(audio, amount=0.35):
    crushed = bus_compressor(audio, threshold=0.10, ratio=8.0, makeup=1.8)
    out = (1 - amount) * audio + amount * crushed
    return clamp(out)


def saturation(audio, drive=1.6, mix=0.35):
    wet = np.tanh(audio * drive)
    out = (1 - mix) * audio + mix * wet
    return clamp(out)


def stereo_widen(audio, width=1.18):
    left = audio[:, 0]
    right = audio[:, 1]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    side *= width

    new_left = mid + side
    new_right = mid - side

    out = np.stack([new_left, new_right], axis=1)
    return clamp(out)


def transient_push(audio, amount=0.12):
    """
    Adds a little attack/punch.
    """
    diff = np.zeros_like(audio)
    diff[1:] = audio[1:] - audio[:-1]
    out = audio + amount * diff
    return clamp(out)


def limiter(audio, ceiling=0.98):
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return clamp(audio)


def final_loudness_push(audio, target_peak=0.98):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio * (target_peak / peak)
    return clamp(audio)


def pro_master(audio):
    audio = eq_stage(audio)
    audio = bus_compressor(audio, threshold=0.24, ratio=3.0, makeup=1.18)
    audio = parallel_compression(audio, amount=0.32)
    audio = transient_push(audio, amount=0.10)
    audio = saturation(audio, drive=1.8, mix=0.30)
    audio = stereo_widen(audio, width=1.16)
    audio = limiter(audio, ceiling=0.97)
    audio = final_loudness_push(audio, target_peak=0.97)
    return clamp(audio)