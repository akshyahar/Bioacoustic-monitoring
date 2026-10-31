import os
import json
import joblib
import numpy as np
import librosa


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/forest_health_mfcc_svm.joblib"

AUDIO_DIR = "dataset/processed_audio"

OUTPUT_DIR = "outputs/survey"

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "forest_health_mfcc_report.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CLASS FOLDERS
# ============================================================

CLASS_PATHS = {
    "Asian_Koel":
        os.path.join(
            AUDIO_DIR,
            "birds",
            "Asian_Koel",
            "birds"
        ),

    "Black_Drongo":
        os.path.join(
            AUDIO_DIR,
            "birds",
            "Black_Drongo",
            "birds"
        ),

    "Indian_Peafowl":
        os.path.join(
            AUDIO_DIR,
            "birds",
            "Indian_Peafowl",
            "birds"
        ),

    "Jungle_Babbler":
        os.path.join(
            AUDIO_DIR,
            "birds",
            "Jungle_Babbler",
            "birds"
        ),

    "White-throated_Kingfisher":
        os.path.join(
            AUDIO_DIR,
            "birds",
            "White-throated_Kingfisher",
            "birds"
        ),

    "SCIALT":
        os.path.join(
            AUDIO_DIR,
            "frogs",
            "SCIALT"
        )
}


# ============================================================
# MFCC EXTRACTION
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

    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)

    return np.concatenate([
        mean,
        std
    ]).reshape(1, -1)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 60)
print("FOREST HEALTH ASSESSMENT")
print("=" * 60)

print()
print("Loading MFCC-SVM model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# INITIALIZE
# ============================================================

species_counts = {
    class_name: 0
    for class_name in CLASS_PATHS
}

total_recordings = 0

confidence_values = []


# ============================================================
# SCAN ALL RECORDINGS
# ============================================================

print()
print("Scanning audio recordings...")
print("-" * 60)


for class_name, folder in CLASS_PATHS.items():

    if not os.path.exists(folder):

        print(
            "WARNING: Folder not found:",
            folder
        )

        continue

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(
            (".wav", ".mp3")
        )
    ]

    for file in files:

        file_path = os.path.join(
            folder,
            file
        )

        try:

            features = extract_mfcc(
                file_path
            )

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )[0]

            confidence = np.max(
                probabilities
            )

            species_counts[prediction] += 1

            confidence_values.append(
                confidence
            )

            total_recordings += 1

        except Exception as e:

            print(
                f"Error processing {file}: {e}"
            )


# ============================================================
# SPECIES DIVERSITY
# ============================================================

species_detected = sum(
    1
    for count in species_counts.values()
    if count > 0
)

total_species = len(
    species_counts
)

diversity_percentage = (
    species_detected /
    total_species
) * 100


# ============================================================
# AVERAGE CONFIDENCE
# ============================================================

if confidence_values:

    average_confidence = (
        np.mean(confidence_values)
        * 100
    )

else:

    average_confidence = 0


# ============================================================
# FOREST HEALTH INDEX
# ============================================================

# Diversity contributes 70%
# Classification confidence contributes 30%

forest_health_score = (
    diversity_percentage * 0.70
    +
    average_confidence * 0.30
)


forest_health_score = min(
    100,
    forest_health_score
)


# ============================================================
# HEALTH STATUS
# ============================================================

if forest_health_score >= 80:

    status = "Healthy"

elif forest_health_score >= 60:

    status = "Moderately Healthy"

elif forest_health_score >= 40:

    status = "Vulnerable"

else:

    status = "Degraded"


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("FOREST HEALTH RESULTS")
print("=" * 60)

print()

print(
    "Total recordings:",
    total_recordings
)

print(
    "Species detected:",
    f"{species_detected} / {total_species}"
)

print(
    "Acoustic diversity:",
    f"{diversity_percentage:.2f}%"
)

print(
    "Average prediction confidence:",
    f"{average_confidence:.2f}%"
)

print(
    "Forest Health Score:",
    f"{forest_health_score:.2f}/100"
)

print(
    "Forest Status:",
    status
)


# ============================================================
# SPECIES DISTRIBUTION
# ============================================================

print()
print("Species Distribution")
print("-" * 60)

for species, count in species_counts.items():

    print(
        f"{species:<30} {count}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report = {

    "model": "MFCC + SVM",

    "total_recordings":
        total_recordings,

    "species_detected":
        species_detected,

    "total_species":
        total_species,

    "acoustic_diversity_percentage":
        round(
            diversity_percentage,
            2
        ),

    "average_prediction_confidence":
        round(
            average_confidence,
            2
        ),

    "forest_health_score":
        round(
            forest_health_score,
            2
        ),

    "forest_status":
        status,

    "species_distribution":
        species_counts
}


with open(
    REPORT_PATH,
    "w"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


print()
print(
    "Report saved to:",
    REPORT_PATH
)

print()
print("=" * 60)