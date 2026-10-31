import os
import sys
import numpy as np
import librosa
import joblib


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/forest_health_mfcc_svm.joblib"


# ============================================================
# MFCC FEATURE EXTRACTION
# ============================================================

def extract_mfcc(file_path):

    y, sr = librosa.load(
        file_path,
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features.reshape(1, -1)


# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("ERROR: MFCC-SVM model not found.")
    print("Expected:", MODEL_PATH)
    sys.exit()


model = joblib.load(MODEL_PATH)


# ============================================================
# GET AUDIO FILE
# ============================================================

if len(sys.argv) < 2:

    print()
    print("Usage:")
    print(
        "python scripts\\predict_mfcc_svm.py "
        "\"path_to_audio.wav\""
    )

    sys.exit()


audio_path = sys.argv[1]


if not os.path.exists(audio_path):

    print("ERROR: Audio file not found.")
    print(audio_path)
    sys.exit()


# ============================================================
# EXTRACT FEATURES
# ============================================================

features = extract_mfcc(audio_path)


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(features)

predicted_class = prediction[0]


# ============================================================
# PROBABILITIES
# ============================================================

probabilities = model.predict_proba(features)[0]

classes = model.classes_


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("FOREST BIOACOUSTIC PREDICTION")
print("=" * 60)

print()
print("Audio file:")
print(os.path.basename(audio_path))

print()
print("Predicted Species:")
print(predicted_class)


print()
print("Class Probabilities:")
print("-" * 60)

for class_name, probability in zip(classes, probabilities):

    print(
        f"{class_name:<30} "
        f"{probability * 100:.2f}%"
    )


confidence = np.max(probabilities)

print()
print(
    f"Confidence: {confidence * 100:.2f}%"
)

print()
print("=" * 60)