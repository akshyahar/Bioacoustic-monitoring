import os
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "dataset/features/cnn_dataset"
MODEL_PATH = "models/forest_health_cnn.keras"
MODEL_DIR = "models"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 8


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained CNN")
print("=" * 60)

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    os.path.join(MODEL_DIR, "class_names.json"),
    "r"
) as f:

    class_names = json.load(f)


print("Classes:")
for i, name in enumerate(class_names):
    print(i, "->", name)


# ============================================================
# LOAD TEST DATA
# ============================================================

test_dir = os.path.join(
    DATASET_DIR,
    "test"
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ============================================================
# GET TRUE LABELS
# ============================================================

y_true = []

for images, labels in test_ds:

    y_true.extend(
        np.argmax(
            labels.numpy(),
            axis=1
        )
    )


y_true = np.array(y_true)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(test_ds)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(cm)

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(class_names)),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("CNN Confusion Matrix")

plt.tight_layout()

cm_path = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "\nConfusion matrix saved to:",
    cm_path
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = np.mean(
    y_true == y_pred
)

print(
    "\nOverall Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print("\nEvaluation completed!")