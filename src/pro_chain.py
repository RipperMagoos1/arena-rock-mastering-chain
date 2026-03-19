import numpy as np


def clamp(audio, low=-1.0, high=1.0):
    return np.clip(audio, low, high)


def highpass_simple(audio, strength=0.995):
    out = np.copy(audio)
    out[1:] = audio[1:] - strength * audio[:-1]
    return out


def lowpass_simple(audio, alpha=0.08):
    out = np.copy(audio)
    for ch in range(audio.shape[1]):
        for i in range(1, len(audio)):
            out[i, ch] = alpha * audio[i, ch] + (1 - alpha) * out[i - 1, ch]
    return out


def add_body(audio, amount=0.16):
    """
    Adds low-mid/body support so the mix feels fuller.
    """
    body = lowpass_simple(audio, alpha=0.025) - lowpass_simple(audio, alpha=0.008)
    return clamp(audio + amount * body)


def sub_bass_support(audio, amount=0.06):
    """
    Light sub/low support without boom.
    """
    lows = lowpass_simple(audio, alpha=0.006)
    return clamp(audio + amount * lows)


def eq_stage(audio):
    """
    Original bright/clear tone idea, but less harsh and with a little more weight.
    """
    rumble_cut = highpass_simple(audio, strength=0.995)
    smooth = lowpass_simple(rumble_cut, alpha=0.18)

    shaped = 0.82 * rumble_cut + 0.18 * smooth

    # original presence idea, slightly reduced
    presence = shaped - lowpass_simple(shaped, alpha=0.04)
    shaped = shaped + 0.05 * presence

    # add fullness back in
    shaped = add_body(shaped, amount=0.14)
    shaped = sub_bass_support(shaped, amount=0.05)

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


def bus_compressor(audio, threshold=0.24, ratio=2.8, makeup=1.15):
    out = np.copy(audio)
    mono = np.mean(np.abs(audio), axis=1)
    env = envelope(mono, attack_coeff=0.2, release_coeff=0.002)

    gain = np.ones_like(env)
    for i in range(len(env)):
        if env[i] > threshold:
            over = env[i] - threshold
            compressed = threshold + over / ratio
            gain[i] = compressed / max(env[i], 1e-9)

    out[:, 0] *= gain
    out[:, 1] *= gain
    out *= makeup
    return clamp(out)


def parallel_compression(audio, amount=0.26):
    crushed = bus_compressor(audio, threshold=0.11, ratio=7.0, makeup=1.45)
    return clamp((1 - amount) * audio + amount * crushed)


def saturation(audio, drive=1.6, mix=0.24):
    wet = np.tanh(audio * drive)
    return clamp((1 - mix) * audio + mix * wet)


def stereo_widen(audio, width=1.10):
    left = audio[:, 0]
    right = audio[:, 1]

    mid = (left + right) * 0.5
    side = (left - right) * 0.5

    side *= width

    new_left = mid + side
    new_right = mid - side

    out = np.stack([new_left, new_right], axis=1)
    return clamp(out)


def transient_push(audio, amount=0.08):
    diff = np.zeros_like(audio)
    diff[1:] = audio[1:] - audio[:-1]
    return clamp(audio + amount * diff)


def limiter(audio, ceiling=0.95):
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return clamp(audio)


def final_trim(audio, gain=0.98):
    return clamp(audio * gain)


def pro_master(audio):
    audio = eq_stage(audio)
    audio = bus_compressor(audio, threshold=0.24, ratio=2.8, makeup=1.12)
    audio = parallel_compression(audio, amount=0.24)
    audio = transient_push(audio, amount=0.07)
    audio = saturation(audio, drive=1.55, mix=0.22)
    audio = stereo_widen(audio, width=1.08)
    audio = limiter(audio, ceiling=0.95)
    audio = final_trim(audio, gain=0.98)
    return clamp(audio)