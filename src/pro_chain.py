import numpy as np


def clamp(audio, low=-1.0, high=1.0):
    return np.clip(audio, low, high)


def lowpass(audio, alpha=0.08):
    out = np.copy(audio)
    for ch in range(audio.shape[1]):
        for i in range(1, len(audio)):
            out[i, ch] = alpha * audio[i, ch] + (1 - alpha) * out[i - 1, ch]
    return out


def highpass_from_lowpass(audio, alpha=0.08):
    return audio - lowpass(audio, alpha=alpha)


def sub_bass_enhancer(audio, amount=0.10):
    """
    Adds a little low-end body without huge boom.
    """
    lows = lowpass(audio, alpha=0.012)
    return clamp(audio + amount * lows)


def mud_control(audio, amount=0.10):
    """
    Pulls back low-mids slightly so the mix stops sounding cloudy.
    """
    low_mid = lowpass(audio, alpha=0.05) - lowpass(audio, alpha=0.015)
    return clamp(audio - amount * low_mid)


def presence_lift(audio, amount=0.12):
    """
    Adds upper-mid clarity for vocals and guitars.
    """
    presence = lowpass(audio, alpha=0.18) - lowpass(audio, alpha=0.06)
    return clamp(audio + amount * presence)


def air_exciter(audio, amount=0.05, drive=2.0):
    """
    Very light harmonic exciter for top-end clarity.
    """
    highs = highpass_from_lowpass(audio, alpha=0.18)
    excited = np.tanh(highs * drive)
    return clamp(audio + amount * excited)


def vocal_lift_curve(audio, amount=0.08):
    """
    Gentle lift centered in the vocal intelligibility zone.
    """
    upper_mids = lowpass(audio, alpha=0.12) - lowpass(audio, alpha=0.035)
    return clamp(audio + amount * upper_mids)


def envelope(signal, attack_coeff=0.18, release_coeff=0.003):
    env = np.zeros_like(signal)
    for i in range(1, len(signal)):
        x = abs(signal[i])
        if x > env[i - 1]:
            env[i] = attack_coeff * x + (1 - attack_coeff) * env[i - 1]
        else:
            env[i] = release_coeff * x + (1 - release_coeff) * env[i - 1]
    return env


def bus_compressor(audio, threshold=0.24, ratio=2.4, makeup=1.06):
    """
    Glue compression. Lower makeup than before so it doesn't feel over-gained.
    """
    out = np.copy(audio)
    mono = np.mean(np.abs(audio), axis=1)
    env = envelope(mono, attack_coeff=0.18, release_coeff=0.003)

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


def parallel_compression(audio, amount=0.18):
    """
    Adds density and punch without getting boxy.
    """
    crushed = bus_compressor(audio, threshold=0.13, ratio=5.5, makeup=1.18)
    return clamp((1 - amount) * audio + amount * crushed)


def transient_punch(audio, amount=0.06):
    """
    Small transient enhancement so drums keep some attack.
    """
    diff = np.zeros_like(audio)
    diff[1:] = audio[1:] - audio[:-1]
    return clamp(audio + amount * diff)


def saturation(audio, drive=1.35, mix=0.18):
    """
    Mild harmonic thickening without harsh fizz.
    """
    wet = np.tanh(audio * drive)
    return clamp((1 - mix) * audio + mix * wet)


def stereo_widen(audio, width=1.06):
    """
    Small width bump only. Too much width smeared clarity before.
    """
    left = audio[:, 0]
    right = audio[:, 1]

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    side *= width

    new_left = mid + side
    new_right = mid - side

    return clamp(np.stack([new_left, new_right], axis=1))


def final_limiter(audio, ceiling=0.94):
    """
    Conservative peak limit so the output doesn't sound overdriven.
    """
    peak = np.max(np.abs(audio))
    if peak > ceiling:
        audio = audio * (ceiling / peak)
    return clamp(audio)


def final_trim(audio, gain=0.97):
    """
    Tiny final trim so it doesn't feel like the gain got pushed too hard.
    """
    return clamp(audio * gain)


def pro_master(audio):
    # Start with weight
    audio = sub_bass_enhancer(audio, amount=0.08)

    # Remove cloudiness
    audio = mud_control(audio, amount=0.11)

    # Glue and punch
    audio = bus_compressor(audio, threshold=0.24, ratio=2.4, makeup=1.05)
    audio = parallel_compression(audio, amount=0.16)
    audio = transient_punch(audio, amount=0.05)

    # Harmonic body
    audio = saturation(audio, drive=1.3, mix=0.16)

    # Clarity stages
    audio = presence_lift(audio, amount=0.10)
    audio = vocal_lift_curve(audio, amount=0.07)
    audio = air_exciter(audio, amount=0.045, drive=1.9)

    # Slight width
    audio = stereo_widen(audio, width=1.05)

    # Safer final level
    audio = final_limiter(audio, ceiling=0.94)
    audio = final_trim(audio, gain=0.97)

    return clamp(audio)