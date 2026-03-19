from src.wav_io import read_wav, write_wav
from src.saturation import soft_clip_saturation
from src.stereo import widen
from src.compression import compressor
from src.utils import clamp

def simple_eq(audio):
    # basic tone shaping (no scipy)
    return audio * 0.98  # subtle smoothing placeholder

def simple_limiter(audio):
    return clamp(audio * 0.9)

if __name__ == "__main__":
    audio, sr = read_wav("input.wav")

    # Chain
    audio = simple_eq(audio)
    audio = compressor(audio, sr)
    audio = soft_clip_saturation(audio, 0.4)
    audio = widen(audio, 0.1)
    audio = simple_limiter(audio)

    write_wav("output_master.wav", audio, sr)

    print("Done: output_master.wav created")