import os
import matplotlib.pyplot as plt
import numpy as np

# Class-wise evaluation results
classes = [
    "Asian Koel",
    "Black Drongo",
    "Indian Peafowl",
    "Jungle Babbler",
    "SCIALT (Frog)",
    "White-throated Kingfisher"
]

precision = [0.75, 0.67, 1.00, 0.40, 1.00, 0.75]
recall =    [0.75, 1.00, 0.50, 0.50, 0.75, 0.75]
f1_score =  [0.75, 0.80, 0.67, 0.44, 0.86, 0.75]

x = np.arange(len(classes))
width = 0.25

# Create output directory
os.makedirs("outputs/figures", exist_ok=True)

plt.figure(figsize=(12, 7))

plt.bar(
    x - width,
    precision,
    width,
    label="Precision"
)

plt.bar(
    x,
    recall,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    f1_score,
    width,
    label="F1-Score"
)

# Add values above bars
for values, positions in [
    (precision, x - width),
    (recall, x),
    (f1_score, x + width)
]:
    for position, value in zip(positions, values):
        plt.text(
            position,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.xlabel("Species / Class")
plt.ylabel("Score")
plt.title(
    "Class-wise Performance of MFCC + RBF-SVM Classifier"
)

plt.xticks(
    x,
    classes,
    rotation=25,
    ha="right"
)

plt.ylim(0, 1.15)

plt.legend()

plt.tight_layout()

output_path = "outputs/figures/classification_performance.png"

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("=" * 60)
print("CLASSIFICATION PERFORMANCE")
print("=" * 60)
print(f"Figure saved to: {output_path}")
print("=" * 60)