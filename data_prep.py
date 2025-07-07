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

NUM_SAMPLES = 8732
SAMPLE_RATE = 16000
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 256
DATASET_NAME = "danavery/urbansound8K"
SPLIT="train"
TARGET_TIME_FRAME = 128

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

    # Handle very short audio clips
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)), mode='constant')

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        window='hann',
        center=True,
        pad_mode='reflect'
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_norm = (mel_db + 80) / 80

    if mel_norm.shape[1] < target_time_frames:
        # Pad with zeros if too short
        pad_width = target_time_frames - mel_norm.shape[1]
        mel_norm = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Truncate if too long
        mel_norm = mel_norm[:, :target_time_frames]

    # Convert to torch tensor
    mel_tensor = torch.tensor(mel_norm).unsqueeze(0).float()

    return {
        "input_tensor": mel_tensor,
        "label": example["classID"],
        "fold": example["fold"]  # Keep the fold information
    }

print(f"Processing {NUM_SAMPLES} samples...")

# Take only the first NUM_SAMPLES
ds_subset = ds.take(NUM_SAMPLES)

# Use dataset.map() for efficient processing
processed_ds = ds_subset.map(
    preprocess,
)

# Convert to list
all_data = list(processed_ds)

# Simple train/val/test split (for non-kfold training)
print("Creating simple train/val/test split...")
total_samples = len(all_data)
train_end = int(total_samples * 0.75)
val_end = train_end + int(total_samples * 0.20)

train_data = all_data[:train_end]
val_data = all_data[train_end:val_end]
test_data = all_data[val_end:]

print(f"Total samples: {total_samples}")
print(f"Train samples: {len(train_data)} ({len(train_data)/total_samples*100:.1f}%)")
print(f"Validation samples: {len(val_data)} ({len(val_data)/total_samples*100:.1f}%)")
print(f"Test samples: {len(test_data)} ({len(test_data)/total_samples*100:.1f}%)")

# Save simple splits
print("Saving train/val/test splits...")
with open("urban_train.pkl", "wb") as f:
    pickle.dump(train_data, f)

with open("urban_val.pkl", "wb") as f:
    pickle.dump(val_data, f)

with open("urban_test.pkl", "wb") as f:
    pickle.dump(test_data, f)

# Also save all data with fold information for k-fold training
print("Saving all data with fold information...")
with open("urban_all_with_folds.pkl", "wb") as f:
    pickle.dump(all_data, f)

print("Done ✅")
sys.exit(0)