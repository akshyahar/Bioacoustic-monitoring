import os
import json
import joblib
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = "dataset/processed_audio"

MODEL_PATH = "models/forest_health_mfcc_svm.joblib"

OUTPUT_DIR = "outputs/evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


CLASS_PATHS = {

    "Asian_Koel":
        os.path.join(
            BASE_DIR,
            "birds",
            "Asian_Koel",
            "birds"
        ),

    "Black_Drongo":
        os.path.join(
            BASE_DIR,
            "birds",
            "Black_Drongo",
            "birds"
        ),

    "Indian_Peafowl":
        os.path.join(
            BASE_DIR,
            "birds",
            "Indian_Peafowl",
            "birds"
        ),

    "Jungle_Babbler":
        os.path.join(
            BASE_DIR,
            "birds",
            "Jungle_Babbler",
            "birds"
        ),

    "White-throated_Kingfisher":
        os.path.join(
            BASE_DIR,
            "birds",
            "White-throated_Kingfisher",
            "birds"
        ),

    "SCIALT":
        os.path.join(
            BASE_DIR,
            "frogs",
            "SCIALT"
        )
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_mfcc(path):

    y, sr = librosa.load(
        path,
        sr=22050,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    mean = np.mean(
        mfcc,
        axis=1
    )

    std = np.std(
        mfcc,
        axis=1
    )

    return np.concatenate([
        mean,
        std
    ])


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

X = []
y = []


for class_name, folder in CLASS_PATHS.items():

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".wav", ".mp3")
        )
    ]

    for file in files:

        path = os.path.join(
            folder,
            file
        )

        try:

            X.append(
                extract_mfcc(path)
            )

            y.append(
                class_name
            )

        except Exception as e:

            print(
                "Error:",
                path,
                e
            )


X = np.array(X)
y = np.array(y)


print()
print(
    "Total samples:",
    len(X)
)

print(
    "Feature size:",
    X.shape[1]
)


# ============================================================
# SAME TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# TRAIN MODEL
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
            probability=True,
            class_weight="balanced"
        )
    )
])


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    y_pred
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

classes = list(
    CLASS_PATHS.keys()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

display.plot(
    ax=ax,
    xticks_rotation=45
)

ax.set_title(
    "MFCC + SVM Confusion Matrix"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "mfcc_svm_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "model":
        "MFCC + RBF-SVM",

    "total_samples":
        len(X),

    "training_samples":
        len(X_train),

    "testing_samples":
        len(X_test),

    "feature_size":
        X.shape[1],

    "test_accuracy":
        round(
            accuracy * 100,
            2
        ),

    "classification_report":
        report,

    "confusion_matrix":
        cm.tolist()
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "mfcc_svm_metrics.json"
)


with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print()

print(
    f"Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)

print()

print(
    "Confusion matrix saved to:"
)

print(
    cm_path
)

print()

print(
    "Metrics saved to:"
)

print(
    metrics_path
)

print()
print("=" * 60)