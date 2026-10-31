import os
import json
import numpy as np
import librosa
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "dataset/processed_audio"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# ACTUAL CLASS FOLDERS IN YOUR PROJECT
# ============================================================

CLASS_PATHS = {
    "Asian_Koel":
        os.path.join(BASE_DIR, "birds", "Asian_Koel", "birds"),

    "Black_Drongo":
        os.path.join(BASE_DIR, "birds", "Black_Drongo", "birds"),

    "Indian_Peafowl":
        os.path.join(BASE_DIR, "birds", "Indian_Peafowl", "birds"),

    "Jungle_Babbler":
        os.path.join(BASE_DIR, "birds", "Jungle_Babbler", "birds"),

    "White-throated_Kingfisher":
        os.path.join(BASE_DIR, "birds", "White-throated_Kingfisher", "birds"),

    "SCIALT":
        os.path.join(BASE_DIR, "frogs", "SCIALT")
}


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

    # Mean + Standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate([
        mfcc_mean,
        mfcc_std
    ])

    return features


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading audio dataset")
print("=" * 60)

X = []
y = []

for class_name, folder in CLASS_PATHS.items():

    if not os.path.exists(folder):
        print(f"WARNING: Folder not found: {folder}")
        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".wav", ".mp3"))
    ]

    print(f"{class_name} : {len(files)} files")

    for file in files:

        file_path = os.path.join(folder, file)

        try:
            features = extract_mfcc(file_path)

            X.append(features)
            y.append(class_name)

        except Exception as e:
            print(f"Error processing {file}: {e}")


X = np.array(X)
y = np.array(y)

print()
print("Total samples:", len(X))
print("Feature size:", X.shape[1])


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# MFCC + SVM
# ============================================================

print()
print("Training MFCC + SVM")
print("=" * 60)

model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "svm",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced"
        )
    )
])


model.fit(X_train, y_train)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print()
print("=" * 60)
print("MFCC + SVM RESULTS")
print("=" * 60)

print()
print("Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

print("Confusion Matrix:")

classes = list(CLASS_PATHS.keys())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "forest_health_mfcc_svm.joblib"
)

joblib.dump(model, model_path)


# Save class names
class_names_path = os.path.join(
    MODEL_DIR,
    "mfcc_class_names.json"
)

with open(class_names_path, "w") as f:
    json.dump(classes, f, indent=4)


print()
print("Model saved to:", model_path)
print("Class names saved to:", class_names_path)

print()
print("MFCC + SVM training completed!")