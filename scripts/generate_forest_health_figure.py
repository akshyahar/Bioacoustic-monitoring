import os
import matplotlib.pyplot as plt


# ============================================================
# FOREST HEALTH INDEX - PROTOTYPE
# ============================================================

# Values obtained from the current 118-recording survey
normalized_shannon = 1.00
evenness = 1.00

# Average ADI across the dataset
# The script will calculate this from the audio files.
average_adi = None

# Average SVM confidence from forest_survey.py
average_confidence = 76.87 / 100


# ============================================================
# Calculate average spectral ADI
# ============================================================

import math
import librosa
import numpy as np


audio_folders = [
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

    stft = librosa.stft(
        y,
        n_fft=2048,
        hop_length=512
    )

    power = np.abs(stft) ** 2

    frequency_energy = np.mean(
        power,
        axis=1
    )

    total_energy = np.sum(frequency_energy)

    if total_energy == 0:
        return 0.0

    probabilities = (
        frequency_energy / total_energy
    )

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    max_entropy = math.log(
        len(frequency_energy)
    )

    return entropy / max_entropy


adi_values = []

for folder in audio_folders:

    if not os.path.exists(folder):
        continue

    for filename in os.listdir(folder):

        if filename.lower().endswith(".wav"):

            filepath = os.path.join(
                folder,
                filename
            )

            try:
                adi_values.append(
                    calculate_adi(filepath)
                )

            except Exception as e:

                print(
                    f"Skipping {filename}: {e}"
                )


average_adi = np.mean(adi_values)


# ============================================================
# Forest Health Index
# ============================================================

health_score = (
    0.30 * normalized_shannon * 100
    + 0.20 * evenness * 100
    + 0.20 * average_adi * 100
    + 0.30 * average_confidence * 100
)


# ============================================================
# Determine prototype status
# ============================================================

if health_score >= 80:
    status = "Healthy"
elif health_score >= 60:
    status = "Moderately Healthy"
elif health_score >= 40:
    status = "Vulnerable"
else:
    status = "Degraded"


# ============================================================
# Display results
# ============================================================

print("=" * 60)
print("FOREST HEALTH INDEX")
print("=" * 60)

print(
    f"Normalized Shannon Diversity : "
    f"{normalized_shannon:.4f}"
)

print(
    f"Shannon Evenness             : "
    f"{evenness:.4f}"
)

print(
    f"Average Spectral ADI         : "
    f"{average_adi:.4f}"
)

print(
    f"Average SVM Confidence       : "
    f"{average_confidence * 100:.2f}%"
)

print("-" * 60)

print(
    f"Forest Health Index          : "
    f"{health_score:.2f}/100"
)

print(
    f"Prototype Status             : "
    f"{status}"
)

print("=" * 60)


# ============================================================
# Create visualization
# ============================================================

os.makedirs(
    "outputs/figures",
    exist_ok=True
)


components = [
    "Shannon\nDiversity",
    "Evenness",
    "Spectral\nADI",
    "SVM\nConfidence"
]

values = [
    normalized_shannon * 100,
    evenness * 100,
    average_adi * 100,
    average_confidence * 100
]

weights = [
    "30%",
    "20%",
    "20%",
    "30%"
]


plt.figure(figsize=(10, 6))

bars = plt.bar(
    components,
    values
)


plt.ylabel("Normalized Score (%)")

plt.xlabel("Forest Health Components")

plt.title(
    "Proposed Forest Health Index Components"
)

plt.ylim(0, 110)


# Add values and weights
for bar, value, weight in zip(
    bars,
    values,
    weights
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%\nWeight: {weight}",
        ha="center",
        va="bottom"
    )


# Display final score in title area
plt.figtext(
    0.5,
    0.01,
    f"Proposed Forest Health Index: "
    f"{health_score:.2f}/100 — {status}",
    ha="center",
    fontsize=11
)


plt.tight_layout(
    rect=[0, 0.05, 1, 1]
)


output_path = (
    "outputs/figures/"
    "forest_health_index.png"
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