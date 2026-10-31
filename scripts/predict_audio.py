import os
import json
import sys
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/forest_health_cnn.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMG_HEIGHT = 128
IMG_WIDTH = 128

SAMPLE_RATE = 22050
DURATION = 3

N_MELS = 128


# ============================================================
# CHECK INPUT
# ============================================================

if len(sys.argv) < 2:
    print("Usage:")
    print("python scripts\\predict_audio.py <audio_file>")
    sys.exit(1)


audio_path = sys.argv[1]

if not os.path.exists(audio_path):
    print("Audio file not found:")
    print(audio_path)
    sys.exit(1)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)


print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# LOAD AUDIO
# ============================================================

print("\nProcessing:")
print(audio_path)

y, sr = librosa.load(
    audio_path,
    sr=SAMPLE_RATE,
    mono=True
)


# ============================================================
# FORCE 3 SECOND AUDIO
# ============================================================

target_samples = SAMPLE_RATE * DURATION

if len(y) < target_samples:

    y = np.pad(
        y,
        (0, target_samples - len(y))
    )

else:

    y = y[:target_samples]


# ============================================================
# CREATE MEL-SPECTROGRAM
# ============================================================

mel = librosa.feature.melspectrogram(
    y=y,
    sr=sr,
    n_mels=N_MELS,
    fmax=sr // 2
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)


# ============================================================
# NORMALIZE
# ============================================================

mel_db = (
    mel_db - mel_db.min()
) / (
    mel_db.max() - mel_db.min() + 1e-8
)


# ============================================================
# CONVERT TO IMAGE
# ============================================================

mel_image = np.expand_dims(
    mel_db,
    axis=-1
)

mel_image = tf.image.resize(
    mel_image,
    [IMG_HEIGHT, IMG_WIDTH]
)

mel_image = np.repeat(
    mel_image.numpy(),
    3,
    axis=-1
)

mel_image = np.expand_dims(
    mel_image,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    mel_image,
    verbose=0
)[0]


predicted_index = np.argmax(
    predictions
)

predicted_class = class_names[
    predicted_index
]

confidence = predictions[
    predicted_index
] * 100


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BIOACOUSTIC CLASSIFICATION RESULT")
print("=" * 60)

print(
    "Predicted class:",
    predicted_class
)

print(
    "Confidence:",
    f"{confidence:.2f}%"
)

print("\nClass probabilities:")

for name, probability in zip(
    class_names,
    predictions
):

    print(
        f"{name:35s} "
        f"{probability * 100:.2f}%"
    )


# ============================================================
# SAVE SPECTROGRAM
# ============================================================

os.makedirs(
    "outputs/predictions",
    exist_ok=True
)

spectrogram_path = (
    "outputs/predictions/latest_melspectrogram.png"
)


plt.figure(
    figsize=(10, 4)
)

librosa.display.specshow(
    mel_db,
    sr=sr,
    x_axis="time",
    y_axis="mel"
)

plt.title(
    f"Predicted: {predicted_class} "
    f"({confidence:.2f}%)"
)

plt.colorbar(
    format="%+2.0f dB"
)

plt.tight_layout()

plt.savefig(
    spectrogram_path,
    dpi=200
)

plt.close()


print(
    "\nSpectrogram saved to:",
    spectrogram_path
)