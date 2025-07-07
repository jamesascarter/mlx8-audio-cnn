import torch
import datasets
from datasets import load_dataset, Audio
import os
import pickle
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import sys

NUM_SAMPLES = 8732  # change to 1000+ for real training
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 512
SAVE_PATH = "urban_subset.pkl"
DATASET_NAME = "danavery/urbansound8K"
SPLIT="train"

ds = datasets.load_dataset(DATASET_NAME, split=SPLIT, streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

def preprocess(example):
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)

    # Resample to 16 kHz if not already
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_norm = (mel_db + 80) / 80  # assuming log scale min ~ -80 dB

    # Convert to torch tensor
    mel_tensor = torch.tensor(mel_norm).unsqueeze(0).float()

    return {
        "input_tensor": mel_tensor,
        "label": example["classID"]  # Use classID instead of class for numeric labels
    }

print(f"Processing {NUM_SAMPLES} samples...")

# Take only the first NUM_SAMPLES
ds_subset = ds.take(NUM_SAMPLES)

# Use dataset.map() for efficient processing
processed_ds = ds_subset.map(
    preprocess,
)

# Convert to list for saving
subset = list(processed_ds)

# Save to file
print(f"Saving {len(subset)} samples to {SAVE_PATH}")
with open(SAVE_PATH, "wb") as f:
    pickle.dump(subset, f)

print("Done ✅")
sys.exit(0)