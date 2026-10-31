import os
import zipfile
import pandas as pd


ZIP_PATH = "dataset/anuraset.zip"
OUTPUT_DIR = "dataset/raw_audio/frogs"
METADATA_OUTPUT = "dataset/metadata/frogs_metadata.csv"

SAMPLES_PER_SPECIES = 20

print("Starting Frog Dataset Extraction")
print("=" * 60)

with zipfile.ZipFile(ZIP_PATH, "r") as z:

    # --------------------------------------------------
    # 1. Load metadata
    # --------------------------------------------------

    with z.open("anuraset/metadata.csv") as f:
        df = pd.read_csv(f)

    print("Metadata loaded")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # --------------------------------------------------
    # 2. Create actual audio filename lookup
    # --------------------------------------------------
    #
    # AnuraSet:
    # sample_name = SAMPLE_00000.wav
    #
    # Actual audio:
    # fname + "_" + min_t + "_" + max_t + ".wav"
    #
    # Example:
    # INCT17_20191113_040000 + 0 + 3
    # =
    # INCT17_20191113_040000_0_3.wav
    # --------------------------------------------------

    wav_paths = [
        x for x in z.namelist()
        if x.lower().endswith(".wav")
    ]

    print("WAV files in ZIP:", len(wav_paths))

    wav_lookup = {
        os.path.basename(path): path
        for path in wav_paths
    }

    # --------------------------------------------------
    # 3. Species columns
    # --------------------------------------------------

    metadata_columns = {
        "sample_name",
        "fname",
        "min_t",
        "max_t",
        "site",
        "date",
        "subset",
        "species_number"
    }

    species_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    print("\nSpecies columns:")
    print(species_columns)

    # --------------------------------------------------
    # 4. Select SINGLE-SPECIES recordings
    # --------------------------------------------------
    #
    # This is important for our CNN classification.
    # We avoid samples containing multiple species.
    # --------------------------------------------------

    single_species_df = df[
        df["species_number"] == 1
    ].copy()

    print(
        "\nSingle-species samples:",
        len(single_species_df)
    )

    selected_records = []

    # --------------------------------------------------
    # 5. Select up to 20 recordings per species
    # --------------------------------------------------

    for species in species_columns:

        species_rows = single_species_df[
            single_species_df[species] == 1
        ].head(SAMPLES_PER_SPECIES)

        found = 0

        for _, row in species_rows.iterrows():

            fname = str(row["fname"])

            min_t = row["min_t"]
            max_t = row["max_t"]

            # Handle integer-like values cleanly
            min_t = int(min_t)
            max_t = int(max_t)

            actual_filename = (
                f"{fname}_{min_t}_{max_t}.wav"
            )

            if actual_filename not in wav_lookup:

                print(
                    "Missing WAV:",
                    actual_filename
                )

                continue

            selected_records.append({
                "species": species,
                "sample_name": row["sample_name"],
                "fname": fname,
                "min_t": min_t,
                "max_t": max_t,
                "site": row["site"],
                "date": row["date"],
                "subset": row["subset"],
                "zip_path": wav_lookup[actual_filename],
                "original_filename": actual_filename
            })

            found += 1

        print(
            f"{species}: "
            f"{found} recordings selected"
        )

    print(
        "\nTotal selected recordings:",
        len(selected_records)
    )

    # --------------------------------------------------
    # 6. Extract recordings
    # --------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    metadata_records = []

    for i, record in enumerate(
        selected_records,
        start=1
    ):

        species = record["species"]

        species_folder = os.path.join(
            OUTPUT_DIR,
            species
        )

        os.makedirs(
            species_folder,
            exist_ok=True
        )

        output_filename = (
            f"{species}_{i:03d}.wav"
        )

        output_path = os.path.join(
            species_folder,
            output_filename
        )

        # Extract audio
        with z.open(
            record["zip_path"]
        ) as source:

            with open(
                output_path,
                "wb"
            ) as target:

                target.write(
                    source.read()
                )

        # Save metadata
        metadata_records.append({
            "Recording_ID": i,
            "Species": species,
            "Original_File": record["original_filename"],
            "Sample_Name": record["sample_name"],
            "Fname": record["fname"],
            "Min_Time": record["min_t"],
            "Max_Time": record["max_t"],
            "Site": record["site"],
            "Date": record["date"],
            "Subset": record["subset"],
            "Local_File": output_path,
            "Source": "AnuraSet"
        })

        print(
            f"[{i}/{len(selected_records)}] "
            f"{species} -> "
            f"{output_filename}"
        )


# --------------------------------------------------
# 7. Save metadata
# --------------------------------------------------

metadata_df = pd.DataFrame(
    metadata_records
)

metadata_df.to_csv(
    METADATA_OUTPUT,
    index=False
)

print("\n" + "=" * 60)
print("Frog Dataset Extraction Completed!")

print(
    "Total recordings:",
    len(metadata_df)
)

print(
    "Metadata saved to:",
    METADATA_OUTPUT
)