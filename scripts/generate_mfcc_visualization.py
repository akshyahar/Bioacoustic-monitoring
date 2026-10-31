import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Input audio
audio_path = (
    "dataset/processed_audio/birds/"
    "Asian_Koel/birds/Asian_Koel_001.wav"
)

# Load audio
y, sr = librosa.load(audio_path, sr=22050)

# Extract 40 MFCCs
mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=40
)

# Create output directory
os.makedirs("outputs/figures", exist_ok=True)

# Create figure
plt.figure(figsize=(12, 7))

librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sr
)

plt.colorbar(label="MFCC Coefficient Value")

plt.xlabel("Time (seconds)")
plt.ylabel("MFCC Coefficient")
plt.title("MFCC Feature Representation of Asian Koel Recording")

plt.tight_layout()

# Save figure
output_path = "outputs/figures/mfcc_visualization.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("MFCC FEATURE VISUALIZATION")
print("=" * 60)
print(f"Audio: {audio_path}")
print(f"Sampling rate: {sr} Hz")
print(f"MFCC coefficients: {mfcc.shape[0]}")
print(f"Output shape: {mfcc.shape}")
print(f"Figure saved to: {output_path}")
print("=" * 60)