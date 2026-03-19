import wave
import numpy as np

def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())

        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        if n_channels == 1:
            audio = np.stack([audio, audio], axis=1)
        else:
            audio = audio.reshape(-1, n_channels)[:, :2]

        return audio, sr


def write_wav(path, audio, sr):
    audio = (audio * 32767).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())