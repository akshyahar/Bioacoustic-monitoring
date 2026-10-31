import os
import json
import numpy as np
import matplotlib.pyplot as plt

metrics_path = "outputs/evaluation/mfcc_svm_metrics.json"
output_dir = "outputs/figures"
output_path = os.path.join(
    output_dir,
    "mfcc_svm_confusion_matrix.png"
)

os.makedirs(output_dir, exist_ok=True)

# Load evaluation metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Get confusion matrix
cm = np.array(metrics["confusion_matrix"])

class_names = [
    "Asian Koel",
    "Black Drongo",
    "Indian Peafowl",
    "Jungle Babbler",
    "SCIALT",
    "White-throated Kingfisher"
]

# Calculate accuracy directly from confusion matrix
accuracy = np.trace(cm) / np.sum(cm)

# Create figure
fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(cm)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(
    "Confusion Matrix of MFCC + RBF-SVM Classifier"
)

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(
    class_names,
    rotation=35,
    ha="right"
)

ax.set_yticklabels(class_names)

# Display values inside matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.colorbar(
    im,
    ax=ax,
    label="Number of Samples"
)

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

print(f"Figure saved to: {output_path}")

print("=" * 60)