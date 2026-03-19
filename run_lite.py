from src.wav_io import read_wav, write_wav
from src.pro_chain import pro_master

if __name__ == "__main__":
    audio, sr = read_wav("input.wav")
    audio = pro_master(audio)
    write_wav("output_master.wav", audio, sr)
    print("Done: output_master.wav created")