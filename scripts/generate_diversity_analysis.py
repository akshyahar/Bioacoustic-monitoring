import os
import math
import librosa
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Dataset species distribution
# ---------------------------------------------------------

species = [
    "Asian Koel",
    "Black Drongo",
    "Indian Peafowl",
    "Jungle Babbler",
    "White-throated Kingfisher",
    "SCIALT (Frog)"
]

counts = np.array([20, 22, 18, 19, 20, 19])

total = counts.sum()

# ---------------------------------------------------------
# Shannon Diversity Index
# ---------------------------------------------------------

proportions = counts / total

shannon_index = -np.sum(
    proportions * np.log(proportions)
)

# Maximum possible Shannon diversity
max_shannon = math.log(len(species))

# Normalized Shannon diversity
normalized_shannon = shannon_index / max_shannon

# Shannon evenness
evenness = shannon_index / max_shannon

# ---------------------------------------------------------
# Calculate spectral ADI for each recording
# ---------------------------------------------------------

audio_paths = [
    "dataset/processed_audio/birds/Asian_Koel/birds",
    "dataset/processed_audio/birds/Black_Drongo/birds",
    "dataset/processed_audio/birds/Indian_Peafowl/birds",
    "dataset/processed_audio/birds/Jungle_Babbler/birds",
    "dataset/processed_audio/birds/White-throated_Kingfisher/birds",
    "dataset/processed_audio/frogs/SCIALT"
]


def calculate_adi(audio_path):

    y, sr = librosa.load(
        audio_path,
        sr=22050,
        mono=True
    )

    # Short-time Fourier transform
    stft = librosa.stft(
        y,
        n_fft=2048,
        hop_length=512
    )

    power = np.abs(stft) ** 2

    # Average energy across time
    frequency_energy = np.mean(
        power,
        axis=1
    )

    total_energy = np.sum(frequency_energy)

    if total_energy == 0:
        return 0.0

    # Convert to energy proportions
    probabilities = (
        frequency_energy / total_energy
    )

    # Remove zero values
    probabilities = probabilities[
        probabilities > 0
    ]

    # Spectral entropy
    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    # Normalize to 0-1
    max_entropy = math.log(len(frequency_energy))

    return entropy / max_entropy


adi_values = []

for folder in audio_paths:

    if not os.path.exists(folder):
        continue

    for filename in os.listdir(folder):

        if filename.lower().endswith(".wav"):

            filepath = os.path.join(
                folder,
                filename
            )

            try:
                adi = calculate_adi(filepath)
                adi_values.append(adi)

            except Exception as e:
                print(
                    f"Could not process {filename}: {e}"
                )

average_adi = np.mean(adi_values)

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("=" * 60)
print("ACOUSTIC DIVERSITY ANALYSIS")
print("=" * 60)

print(f"Total recordings: {total}")
print(f"Species richness: {len(species)}")
print(f"Shannon Diversity Index: {shannon_index:.4f}")
print(f"Normalized Shannon Diversity: {normalized_shannon:.4f}")
print(f"Shannon Evenness: {evenness:.4f}")
print(f"Average Spectral ADI: {average_adi:.4f}")

print("=" * 60)

# ---------------------------------------------------------
# Create figure
# ---------------------------------------------------------

os.makedirs(
    "outputs/figures",
    exist_ok=True
)

metrics = [
    "Normalized\nShannon",
    "Evenness",
    "Spectral ADI"
]

values = [
    normalized_shannon,
    evenness,
    average_adi
]

plt.figure(figsize=(9, 6))

bars = plt.bar(
    metrics,
    values
)

plt.ylabel("Normalized Value")
plt.xlabel("Diversity Metric")

plt.title(
    "Acoustic Diversity Analysis of the Bioacoustic Dataset"
)

plt.ylim(0, 1.1)

# Add values
for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.025,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output_path = (
    "outputs/figures/"
    "acoustic_diversity_analysis.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    f"Figure saved to: {output_path}"
)