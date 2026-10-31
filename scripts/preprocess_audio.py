import os
import librosa
import soundfile as sf


RAW_DIR = "dataset/raw_audio"
PROCESSED_DIR = "dataset/processed_audio"

TARGET_SR = 22050


def process_audio(input_path, output_path):
    """
    Convert audio to:
    - Mono
    - 22050 Hz
    - WAV
    """

    y, sr = librosa.load(
        input_path,
        sr=TARGET_SR,
        mono=True
    )

    os.makedirs(
        os.path.dirname(output_path),
        exist_ok=True
    )

    sf.write(
        output_path,
        y,
        TARGET_SR
    )


def process_folder(input_dir, output_dir):
    """
    Process every audio file inside a folder.
    """

    processed = 0

    for root, dirs, files in os.walk(input_dir):

        for filename in files:

            if not filename.lower().endswith(
                (".wav", ".mp3", ".flac", ".ogg")
            ):
                continue

            input_path = os.path.join(
                root,
                filename
            )

            relative_path = os.path.relpath(
                input_path,
                input_dir
            )

            output_path = os.path.join(
                output_dir,
                relative_path
            )

            output_path = (
                os.path.splitext(output_path)[0]
                + ".wav"
            )

            print(f"Processing: {input_path}")

            try:

                process_audio(
                    input_path,
                    output_path
                )

                processed += 1

            except Exception as e:

                print(f"ERROR: {input_path}")
                print(e)

    return processed


print("Starting Audio Preprocessing")
print("=" * 60)


# --------------------------------------------------
# BIRDS
# --------------------------------------------------

print("\nProcessing BIRDS")

bird_species = [
    "Asian Koel",
    "Black Drongo",
    "Indian Peafowl",
    "White-throated Kingfisher",
    "Jungle Babbler"
]

bird_total = 0

for species in bird_species:

    input_dir = os.path.join(
        RAW_DIR,
        species
    )

    output_dir = os.path.join(
        PROCESSED_DIR,
        "birds",
        species.replace(" ", "_")
    )

    if os.path.exists(input_dir):

        count = process_folder(
            input_dir,
            output_dir
        )

        print(
            f"{species}: {count} files processed"
        )

        bird_total += count

    else:

        print(
            f"{species}: folder not found"
        )


# --------------------------------------------------
# FROGS
# --------------------------------------------------

print("\nProcessing FROGS")

frog_input = os.path.join(
    RAW_DIR,
    "frogs"
)

frog_output = os.path.join(
    PROCESSED_DIR,
    "frogs"
)

frog_total = process_folder(
    frog_input,
    frog_output
)

print(
    f"Frogs: {frog_total} files processed"
)


# --------------------------------------------------
# INSECTS
# --------------------------------------------------

print("\nProcessing INSECTS")

insect_input = os.path.join(
    RAW_DIR,
    "insects"
)

insect_output = os.path.join(
    PROCESSED_DIR,
    "insects"
)

if os.path.exists(insect_input):

    insect_total = process_folder(
        insect_input,
        insect_output
    )

else:

    insect_total = 0

print(
    f"Insects: {insect_total} files processed"
)


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)

print("Audio preprocessing completed!")

print(
    "Bird files:",
    bird_total
)

print(
    "Frog files:",
    frog_total
)

print(
    "Insect files:",
    insect_total
)

print(
    "Total files:",
    bird_total + frog_total + insect_total
)

print(
    "Processed audio location:",
    PROCESSED_DIR
)