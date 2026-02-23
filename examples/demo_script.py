from src.pipeline import arena_master

# Example:
#   python examples/demo_script.py
# Make sure you have an input.wav in the repo root (or change the path).
if __name__ == "__main__":
    arena_master(
        input_wav="input.wav",
        output_wav="output_master.wav",
        preset="broadcast_master"
    )
    print("Done: wrote output_master.wav")