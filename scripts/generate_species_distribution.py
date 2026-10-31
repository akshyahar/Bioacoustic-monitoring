import os
import matplotlib.pyplot as plt

species = [
    "Asian Koel",
    "Black Drongo",
    "Indian Peafowl",
    "Jungle Babbler",
    "White-throated Kingfisher",
    "SCIALT (Frog)"
]

counts = [20, 22, 18, 19, 20, 19]

os.makedirs("outputs/figures", exist_ok=True)

plt.figure(figsize=(10, 6))

bars = plt.bar(species, counts)

plt.xlabel("Species / Class")
plt.ylabel("Number of Recordings")
plt.title("Species Distribution in the Bioacoustic Dataset")

plt.xticks(rotation=25, ha="right")

for bar, count in zip(bars, counts):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        count + 0.3,
        str(count),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output_path = "outputs/figures/species_distribution.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("=" * 60)
print("SPECIES DISTRIBUTION FIGURE")
print("=" * 60)
print(f"Figure saved to: {output_path}")