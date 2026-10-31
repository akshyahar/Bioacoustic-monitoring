import os
import json
import numpy as np
import librosa
import tensorflow as tf


MODEL_PATH = "models/forest_health_cnn.keras"
CLASS_NAMES_PATH = "models/class_names.json"

SAMPLE_RATE = 22050
DURATION = 3
N_MELS = 128

# Number of recordings to analyze
MAX_FILES = 100


def load_model_and_classes():
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    return model, class_names


def audio_to_mel(audio_path):
    y, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    target_samples = SAMPLE_RATE * DURATION

    if len(y) < target_samples:
        y = np.pad(
            y,
            (0, target_samples - len(y))
        )
    else:
        y = y[:target_samples]

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

    mel_db = (
        mel_db - mel_db.min()
    ) / (
        mel_db.max() - mel_db.min() + 1e-8
    )

    image = np.expand_dims(
        mel_db,
        axis=-1
    )

    image = tf.image.resize(
        image,
        [128, 128]
    )

    image = np.repeat(
        image.numpy(),
        3,
        axis=-1
    )

    return np.expand_dims(image, axis=0)


def analyze_audio_folder(audio_folder):

    model, class_names = load_model_and_classes()

    audio_files = []

    for root, _, files in os.walk(audio_folder):

        for file in files:

            if file.lower().endswith(
                (".wav", ".mp3", ".flac")
            ):

                audio_files.append(
                    os.path.join(root, file)
                )

    audio_files = audio_files[:MAX_FILES]

    print("=" * 60)
    print("FOREST HEALTH AUDIO ANALYSIS")
    print("=" * 60)

    print("Audio files:", len(audio_files))

    predictions = []

    for i, audio_path in enumerate(audio_files, 1):

        print(
            f"[{i}/{len(audio_files)}] "
            f"{os.path.basename(audio_path)}"
        )

        try:

            mel = audio_to_mel(audio_path)

            probability = model.predict(
                mel,
                verbose=0
            )[0]

            predicted_index = np.argmax(
                probability
            )

            predicted_class = class_names[
                predicted_index
            ]

            confidence = probability[
                predicted_index
            ]

            predictions.append({
                "file": audio_path,
                "class": predicted_class,
                "confidence": float(confidence)
            })

        except Exception as e:

            print(
                "ERROR:",
                audio_path,
                e
            )

    return predictions


def calculate_health_score(predictions):

    if not predictions:
        return 0, {}

    class_counts = {}

    for item in predictions:

        species = item["class"]

        class_counts[species] = (
            class_counts.get(species, 0) + 1
        )

    total = len(predictions)

    # Number of classes detected
    detected_species = len(class_counts)

    # Species diversity component
    diversity_score = (
        detected_species / 6
    ) * 100

    # Average confidence
    average_confidence = (
        np.mean([
            item["confidence"]
            for item in predictions
        ]) * 100
    )

    # Final health score
    health_score = (
        0.6 * diversity_score +
        0.4 * average_confidence
    )

    if health_score >= 75:
        status = "Healthy"

    elif health_score >= 50:
        status = "Moderate"

    else:
        status = "Poor"

    metrics = {
        "total_recordings": total,
        "detected_species": detected_species,
        "species_counts": class_counts,
        "diversity_score": round(
            diversity_score, 2
        ),
        "average_confidence": round(
            average_confidence, 2
        ),
        "health_score": round(
            health_score, 2
        ),
        "status": status
    }

    return health_score, metrics


if __name__ == "__main__":

    AUDIO_FOLDER = "dataset/processed_audio"

    predictions = analyze_audio_folder(
        AUDIO_FOLDER
    )

    health_score, metrics = calculate_health_score(
        predictions
    )

    print("\n" + "=" * 60)
    print("FOREST HEALTH RESULT")
    print("=" * 60)

    print(
        "Total recordings:",
        metrics["total_recordings"]
    )

    print(
        "Detected species:",
        metrics["detected_species"]
    )

    print(
        "Species distribution:"
    )

    for species, count in metrics[
        "species_counts"
    ].items():

        print(
            f"  {species}: {count}"
        )

    print(
        "\nDiversity Score:",
        metrics["diversity_score"],
        "%"
    )

    print(
        "Average Confidence:",
        metrics["average_confidence"],
        "%"
    )

    print(
        "\nFOREST HEALTH SCORE:",
        metrics["health_score"],
        "/ 100"
    )

    print(
        "STATUS:",
        metrics["status"]
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    with open(
        "outputs/forest_health_result.json",
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4
        )

    print(
        "\nResult saved to:"
        " outputs/forest_health_result.json"
    )