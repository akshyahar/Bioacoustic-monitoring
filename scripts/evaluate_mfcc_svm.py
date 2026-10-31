import os
import numpy as np
import librosa

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


BASE_DIR = "dataset/processed_audio"


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

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    return np.concatenate([mean, std])


X = []
y = []


print("Loading dataset...")
print("=" * 60)


for class_name, folder in CLASS_PATHS.items():

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".wav", ".mp3"))
    ]

    for file in files:

        path = os.path.join(folder, file)

        try:
            features = extract_mfcc(path)

            X.append(features)
            y.append(class_name)

        except Exception as e:
            print("Error:", path, e)


X = np.array(X)
y = np.array(y)


print("Total samples:", len(X))
print("Feature size:", X.shape[1])


# ============================================================
# MODEL
# ============================================================

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
            class_weight="balanced"
        )
    )
])


# ============================================================
# 5-FOLD CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)


print()
print("=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)

for i, score in enumerate(scores, 1):

    print(
        f"Fold {i}: {score * 100:.2f}%"
    )


print()
print(
    f"Mean Accuracy: {scores.mean() * 100:.2f}%"
)

print(
    f"Standard Deviation: {scores.std() * 100:.2f}%"
)